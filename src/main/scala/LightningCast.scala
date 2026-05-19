import spatial.dsl._

@spatial object LightningCast extends SpatialApp {
  
  type T = Int
  
  def conv2d_layer(
    input: SRAM3[T],
    weights: SRAM4[T],
    bias: SRAM1[T],
    H: scala.Int, W: scala.Int, IC: scala.Int, OC: scala.Int
  ): SRAM3[T] = {
    
    val output = SRAM[T](H, W, OC)
    val K = 3
    val pad = 1
    
    val padded = SRAM[T](H + 2, W + 2, IC)
    
    Foreach(0 until H, 0 until W, 0 until IC) { (h, w, c) =>
      padded(h + pad, w + pad, c) = input(h, w, c)
    }
    
    Foreach(0 until OC par 2) { oc =>
      Foreach(0 until H, 0 until W par 2) { (oh, ow) =>
        val acc = Reduce(Reg[T](0))(0 until K, 0 until K, 0 until IC par 2) { 
          (kh, kw, ic) =>
            padded(oh + kh, ow + kw, ic) * weights(kh, kw, ic, oc)
        } { _ + _ }
        
        val result = acc.value + bias(oc)
        output(oh, ow, oc) = max(0.to[T], result)
      }
    }
    output
  }
  
  def main(args: Array[String]): Unit = {
    
    val H = 64
    val W = 64
    val IC = 4
    val OC = 16
    
    val input_data = Array.tabulate(H * W * IC) { i => (i % 256).to[T] }
    val weight_data = Array.tabulate(3 * 3 * IC * OC) { i => 1.to[T] }
    val bias_data = Array.tabulate(OC) { i => 0.to[T] }
    
    val input_dram = DRAM[T](H, W, IC)
    val weight_dram = DRAM[T](3, 3, IC, OC)
    val bias_dram = DRAM[T](OC)
    val output_dram = DRAM[T](H, W, OC)
    
    setMem(input_dram, input_data)
    setMem(weight_dram, weight_data)
    setMem(bias_dram, bias_data)
    
    Accel {
      val input_sram = SRAM[T](H, W, IC)
      val weight_sram = SRAM[T](3, 3, IC, OC)
      val bias_sram = SRAM[T](OC)
      
      input_sram load input_dram
      weight_sram load weight_dram
      bias_sram load bias_dram
      
      val result = conv2d_layer(input_sram, weight_sram, bias_sram, H, W, IC, OC)
      
      output_dram store result
    }
    
    val output = getMem(output_dram)
    
    println(s"Conv2D complete! Output size: ${output.length}")
    //println(s"Sample outputs: ${output.slice(0, 10).mkString(", ")}")
  }
}