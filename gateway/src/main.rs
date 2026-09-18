//! The transport gateway.
//!
//! One process per machine sits between an engine and the network (Section 5).
//! On the client's machine it reads hidden states and verdicts from a shared
//! memory ring, in stream order, and carries them as protocol-buffer frames
//! over one persistent gRPC connection; outputs come back on the same stream
//! and go into a second ring. On the provider's machine it does the mirror
//! image, serving that connection.
//!
//! The gateway keeps the two traffic classes apart the way Section 4.4
//! requires: it takes one frame at a time from the ring, so a first-segment
//! hidden state waits behind at most one background hidden state, and it never
//! reorders what the scheduler ordered.
//!
//!     slipstream-gateway client   --up up.ring --down down.ring --connect host:50151
//!     slipstream-gateway provider --up up.ring --down down.ring --listen 0.0.0.0:50151

mod frame;
mod ring;
mod tls;

pub mod pb {
    tonic::include_proto!("slipstream.v1");
}

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ring::{Ring, KIND_OUTPUT};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint, Server};
use tonic::{Request, Response, Status, Streaming};
use tracing::{debug, info, warn};

use pb::middle_client::MiddleClient;
use pb::middle_server::{Middle, MiddleServer};

#[derive(Parser, Debug)]
#[command(
    name = "slipstream-gateway",
    about = "Shared memory on one side, protocol buffers over gRPC on the other"
)]
struct Cli {
    #[command(subcommand)]
    role: Role,

    /// Ring the engine writes and the gateway reads.
    #[arg(long, global = true, default_value = ".shm/up.ring")]
    up: PathBuf,

    /// Ring the gateway writes and the engine reads.
    #[arg(long, global = true, default_value = ".shm/down.ring")]
    down: PathBuf,

    /// Slots in a ring the gateway creates.
    #[arg(long, global = true, default_value_t = 1024)]
    slots: usize,

    /// Bytes per slot, which bounds one hidden state.
    #[arg(long, global = true, default_value_t = 65536)]
    slot_bytes: usize,

    /// Create the rings rather than expecting the engine to have made them.
    #[arg(long, global = true)]
    create_rings: bool,

    /// How long to sleep when a ring is empty.
    #[arg(long, global = true, default_value_t = 50)]
    idle_micros: u64,

    #[arg(long, global = true)]
    tls_ca: Option<PathBuf>,
    #[arg(long, global = true)]
    tls_cert: Option<PathBuf>,
    #[arg(long, global = true)]
    tls_key: Option<PathBuf>,
    #[arg(long, global = true)]
    tls_domain: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Role {
    /// The client's side: read the up ring, connect out, fill the down ring.
    Client {
        /// Address of the provider's gateway.
        #[arg(long, default_value = "127.0.0.1:50151")]
        connect: String,
    },
    /// The provider's side: serve the connection, fill the up ring, read the
    /// down ring.
    Provider {
        #[arg(long, default_value = "0.0.0.0:50151")]
        listen: String,
        /// Window the provider announces with every output, when the engine
        /// does not set one itself.
        #[arg(long, default_value_t = 24)]
        window: u32,
    },
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn tls_paths(cli: &Cli) -> tls::TlsPaths {
    tls::TlsPaths {
        ca: cli.tls_ca.clone(),
        cert: cli.tls_cert.clone(),
        key: cli.tls_key.clone(),
        domain: cli.tls_domain.clone(),
    }
}

fn open_ring(path: &PathBuf, create: bool, slots: usize, slot_bytes: usize) -> Result<Ring> {
    if create {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        Ring::create(path, slots, slot_bytes)
    } else {
        Ring::open(path)
    }
}

// ---------------------------------------------------------------- the client

async fn run_client(cli: &Cli, connect: &str) -> Result<()> {
    let mut up = open_ring(&cli.up, cli.create_rings, cli.slots, cli.slot_bytes)?;
    let down = Arc::new(Mutex::new(open_ring(
        &cli.down,
        cli.create_rings,
        cli.slots,
        cli.slot_bytes,
    )?));

    let endpoint = format!("http://{connect}");
    let mut channel = Endpoint::from_shared(endpoint.clone())?
        .tcp_nodelay(true)
        .keep_alive_while_idle(true);
    if let Some(config) = tls_paths(cli).client()? {
        channel = channel.tls_config(config)?;
    }
    let channel: Channel = channel.connect().await.with_context(|| {
        format!("connecting to the provider's gateway at {endpoint}")
    })?;
    let mut client = MiddleClient::new(channel);
    info!(%connect, "one persistent connection, protocol buffers over gRPC");

    // One frame at a time: the channel takes what the scheduler ordered, in
    // that order, and nothing else.
    let (tx, rx) = mpsc::channel::<pb::Upstream>(1);
    let response = client.run(Request::new(ReceiverStream::new(rx))).await?;
    let mut inbound: Streaming<pb::Downstream> = response.into_inner();

    let idle = Duration::from_micros(cli.idle_micros);
    let downstream = down.clone();
    let reader = tokio::spawn(async move {
        while let Some(message) = inbound.next().await {
            let message = match message {
                Ok(m) => m,
                Err(status) => {
                    warn!(%status, "the provider's gateway closed the stream");
                    break;
                }
            };
            if let Some(pb::downstream::Frame::Output(output)) = message.frame {
                let record = frame::output_to_record(&output);
                let mut ring = downstream.lock().await;
                while !ring.push(&record).unwrap_or(false) {
                    tokio::time::sleep(idle).await;
                }
            }
        }
    });

    loop {
        match up.pop()? {
            Some(record) => {
                if let Some(upstream) = frame::record_to_upstream(&record, now_ns()) {
                    if tx.send(upstream).await.is_err() {
                        break;
                    }
                    debug!(position = record.position, "sent one frame");
                }
            }
            None => tokio::time::sleep(idle).await,
        }
    }
    reader.abort();
    Ok(())
}

// -------------------------------------------------------------- the provider

struct MiddleService {
    up: Arc<Mutex<Ring>>,
    down: Arc<Mutex<Ring>>,
    idle: Duration,
    window: u32,
}

#[tonic::async_trait]
impl Middle for MiddleService {
    type RunStream = ReceiverStream<Result<pb::Downstream, Status>>;

    async fn run(
        &self,
        request: Request<Streaming<pb::Upstream>>,
    ) -> Result<Response<Self::RunStream>, Status> {
        let mut inbound = request.into_inner();
        let up = self.up.clone();
        let down = self.down.clone();
        let idle = self.idle;
        let window = self.window;

        // Arrivals go into the up ring for the engine to run.
        tokio::spawn(async move {
            let mut received: u64 = 0;
            while let Some(message) = inbound.next().await {
                let message = match message {
                    Ok(m) => m,
                    Err(status) => {
                        warn!(%status, "a client's gateway closed the stream");
                        break;
                    }
                };
                let record = match message.frame {
                    Some(pb::upstream::Frame::HiddenState(state)) => {
                        received += 1;
                        frame::hidden_state_to_record(&state)
                    }
                    Some(pb::upstream::Frame::Verdict(verdict)) => {
                        frame::verdict_to_record(&verdict)
                    }
                    None => continue,
                };
                let mut ring = up.lock().await;
                while !ring.push(&record).unwrap_or(false) {
                    tokio::time::sleep(idle).await;
                }
            }
            debug!(received, "stream ended");
        });

        // Outputs the engine produces go back down the same stream.
        let (tx, rx) = mpsc::channel::<Result<pb::Downstream, Status>>(16);
        tokio::spawn(async move {
            let mut sent: u64 = 0;
            loop {
                let record = {
                    let mut ring = down.lock().await;
                    ring.pop().unwrap_or(None)
                };
                match record {
                    Some(record) if record.kind == KIND_OUTPUT => {
                        sent += 1;
                        let announced = if record.parent > 0 { record.parent } else { window };
                        let output = frame::record_to_output(&record, announced, sent, now_ns());
                        let message = pb::Downstream {
                            frame: Some(pb::downstream::Frame::Output(output)),
                        };
                        if tx.send(Ok(message)).await.is_err() {
                            break;
                        }
                    }
                    Some(other) => debug!(kind = other.kind, "ignoring a non-output record"),
                    None => tokio::time::sleep(idle).await,
                }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

async fn run_provider(cli: &Cli, listen: &str, window: u32) -> Result<()> {
    let service = MiddleService {
        up: Arc::new(Mutex::new(open_ring(
            &cli.up,
            cli.create_rings,
            cli.slots,
            cli.slot_bytes,
        )?)),
        down: Arc::new(Mutex::new(open_ring(
            &cli.down,
            cli.create_rings,
            cli.slots,
            cli.slot_bytes,
        )?)),
        idle: Duration::from_micros(cli.idle_micros),
        window,
    };
    let address = listen.parse().with_context(|| format!("parsing {listen}"))?;
    let mut server = Server::builder().tcp_nodelay(true);
    if let Some(config) = tls_paths(cli).server()? {
        server = server.tls_config(config)?;
        info!("mutual TLS is on");
    }
    info!(%listen, "serving the middle");
    server
        .add_service(MiddleServer::new(service))
        .serve(address)
        .await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "slipstream_gateway=info".into()),
        )
        .init();
    let cli = Cli::parse();
    match &cli.role {
        Role::Client { connect } => run_client(&cli, connect).await,
        Role::Provider { listen, window } => run_provider(&cli, listen, *window).await,
    }
}
