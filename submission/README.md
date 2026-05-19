**Architecture:**
- Reusable Conv2D module supporting arbitrary channel counts
- 3×3 kernel convolution with padding
- Parallelization factors: `par 2` for output channels, `par 2` for spatial dimensions
- LeakyReLU activation (with clipping)
- MaxPool2D module for downsampling

**Test Configuration:**
- Input: 64×64×4 (H×W×C)
- Weights: 3×3×4×16 (Kh×Kw×IC×OC)
- Output: 64×64×16
- Total operations: ~1.8M MACs
**Simulation artifacts:**
- IR reports in `simulation_reports/`
- Banking decisions in `banking/`
- Detailed controller hierarchy generated

### 3. Design Decisions

**Memory Hierarchy:**
- DRAM for off-chip storage
- SRAM for on-chip buffers (input, weights, output)
- Padding handled in separate SRAM

**Parallelization Strategy:**
- Output channel parallelism: reduces latency by processing multiple filters simultaneously
- Spatial parallelism: processes multiple output pixels in parallel
- Input channel reduction: parallelized accumulation across input channels

**Next Steps (Milestone 3):**
- Increase tile size to 256×256
- Chain multiple layers (encoder → bottleneck → decoder)
- Load real model weights from .npy files
- Optimize parallelization factors for F2 FPGA resources
