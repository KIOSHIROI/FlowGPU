# Security policy

Report vulnerabilities privately to the maintainers via GitHub Security Advisories. Do not open a public issue for an unfixed vulnerability.

FlowGPU compiles source and optionally runs generated Triton kernels. Treat untrusted compiler input as untrusted code; the reference/CPU backend is the safe default and device execution requires an explicit flag.
