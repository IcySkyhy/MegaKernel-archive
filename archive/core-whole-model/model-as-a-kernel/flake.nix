{
  description = "Flake for model-as-a-kernel";

  inputs = {
    builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      builder,
    }:
    builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
