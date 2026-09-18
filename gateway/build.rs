fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The gateway speaks exactly the protocol in ../proto/slipstream.proto.
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["../proto/slipstream.proto"], &["../proto"])?;
    println!("cargo:rerun-if-changed=../proto/slipstream.proto");
    Ok(())
}
