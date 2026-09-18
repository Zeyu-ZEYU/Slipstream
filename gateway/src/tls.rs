//! Mutual TLS, when the deployment asks for it.
//!
//! Section 5: the gateways carry frames "over one persistent gRPC connection
//! per client, with mutual TLS available". Certificates are the deployment's;
//! `deploy/tls/gen_certs.sh` makes a self-signed set for a functional run.

use anyhow::{Context, Result};
use std::path::PathBuf;
use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

#[derive(Debug, Clone, Default)]
pub struct TlsPaths {
    pub ca: Option<PathBuf>,
    pub cert: Option<PathBuf>,
    pub key: Option<PathBuf>,
    pub domain: Option<String>,
}

impl TlsPaths {
    pub fn enabled(&self) -> bool {
        self.ca.is_some() && self.cert.is_some() && self.key.is_some()
    }

    fn identity(&self) -> Result<Identity> {
        let cert = std::fs::read(self.cert.as_ref().context("missing certificate")?)?;
        let key = std::fs::read(self.key.as_ref().context("missing key")?)?;
        Ok(Identity::from_pem(cert, key))
    }

    fn authority(&self) -> Result<Certificate> {
        let ca = std::fs::read(self.ca.as_ref().context("missing certificate authority")?)?;
        Ok(Certificate::from_pem(ca))
    }

    /// Server side: present an identity and require the client to present one.
    pub fn server(&self) -> Result<Option<ServerTlsConfig>> {
        if !self.enabled() {
            return Ok(None);
        }
        Ok(Some(
            ServerTlsConfig::new()
                .identity(self.identity()?)
                .client_ca_root(self.authority()?),
        ))
    }

    /// Client side: verify the server and present an identity.
    pub fn client(&self) -> Result<Option<ClientTlsConfig>> {
        if !self.enabled() {
            return Ok(None);
        }
        let mut config = ClientTlsConfig::new()
            .ca_certificate(self.authority()?)
            .identity(self.identity()?);
        if let Some(domain) = &self.domain {
            config = config.domain_name(domain.clone());
        }
        Ok(Some(config))
    }
}
