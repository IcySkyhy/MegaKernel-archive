# M13 dump manifest（M13_DUMP=1 落盘的中间张量：sha256 全 64 位 + 字节数）

口径：140 个 `.bin` dump 张量（每 case×mode 35 个 × 4 组合）；同目录另有 4 个 `*_meta.txt`，共 144 个文件。
本清单与 `accept_run_m1_m33.log` 出自同一次运行（日志内 4 行 dump 提示）。

生成方式（在空目录里跑，避免污染仓库）：
```
mkdir -p /tmp/m13_dump && cd /tmp/m13_dump
M13_DUMP=1 <repo>/m13_moe_layer/build/m13_moe_layer <repo>/tools/golden/data m1 m33
sha256sum *_device.bin   # 与本表逐行比对（kernel 确定性：两次运行逐字节一致）
```

**M50 复跑（Extract→VF 内联后）**：下表的 140 个张量 sha256 **全部不变**（脚本逐行比对：条目 140、
不符 0、缺失 0；本次落盘的 144 个文件按下表外另有 4 个 `*_meta.txt`）。原始读数为
`evidence/m50_dump_sha256.txt`，比对过程见 `evidence/m50_zero_regression.log` §2/§3。

| 文件 | 字节 | sha256 |
|---|---|---|
| m1_mode0_a_qx_device.bin | 327680 | 90c586780eeb4f4236a3128c007cce770b793450deb075f39e84d7110a33eecf |
| m1_mode0_a_qx_shd_device.bin | 81920 | b611fb500460702b76bc6b84726b3554797846db9bb4a76cd863a71d886af3f7 |
| m1_mode0_a_scale_device.bin | 20480 | 353f0d946ebea093e63c0a9d434b436b20411bcb065adf7f7171522209419b75 |
| m1_mode0_a_scale_shd_device.bin | 5120 | 5cf915e42005f8c39d616cd08069a6b0cf9e754908875c29ce60bf4f1b6cbaba |
| m1_mode0_expert_offsets_device.bin | 64 | 162fe323c55cfd29fc7f8f2f8fea855f85bc3a27d114ded9774eee27c10bdf22 |
| m1_mode0_expert_token_counts_device.bin | 32 | 39169e9bced8a9aba2def574d5fc6b40fefe8213933435a585a5093d9fa6133e |
| m1_mode0_gamma1_input_device.bin | 5120 | d8752cad0f660495869b00205924c25e9819f37c23b34e5834432a05bdad5028 |
| m1_mode0_gamma2_input_device.bin | 5120 | 4ee21fee290fc03af05909ae9fc291894e30b9a577d5a9d60f94e36b6d865507 |
| m1_mode0_gu_device.bin | 655360 | 05cd559434e875bbc2162a10f052d98e1928259d9b36b01bf7ebd2e438bc36f2 |
| m1_mode0_gu_shd_device.bin | 163840 | 275ddb82a158a1f456b37dca24e3577530b8f77870fc511e4c3ba1047f4f58a6 |
| m1_mode0_h_qx_device.bin | 81920 | b704eb2d5d3dfa8d29c6a296d5df0a7ba7cdef136393a1981e34e3d872fcf501 |
| m1_mode0_h_qx_shd_device.bin | 20480 | 965f535b60f2da50f8babeabf50489487edcc68eaa4e685101939129abede210 |
| m1_mode0_h_scale_device.bin | 8192 | 4ee258ed8fcf808c8f5423303d2d4f35920ce9fe3bda9058d09a598ec947130e |
| m1_mode0_h_scale_shd_device.bin | 2048 | 6144a852568927bd227cdcb80148e794c05ccf7c0b059116b6531ce472fea30c |
| m1_mode0_h_swiglu_device.bin | 327680 | bfeb1238f2ff56cb356cbbea9e9b0c06d0f3302a6c97bb0daf844b37eb148286 |
| m1_mode0_h_swiglu_shd_device.bin | 81920 | 3ca34764c9c87d544f408618cdbe06aafe6c23348770057417f395a39aeae859 |
| m1_mode0_ids_device.bin | 1024 | d55db534043933c71af28e18e91f2db213fd359837b3cb9f29b0c3447fed841f |
| m1_mode0_inv_slot_device.bin | 1024 | f1c0ceb07394f25a70a24317f4846a3ae4731392ffae2af84aa1477b11a5c31e |
| m1_mode0_logits_device.bin | 1024 | ff023076f641886ac80903a295c1f86c2c8e695584b029f8f935bbac94750526 |
| m1_mode0_moe_output_device.bin | 327680 | 68e031d2e15b8c9fa74421bad3abc1d5a5f4966bbe704f6ea75d84b02b123c98 |
| m1_mode0_perm_expert_device.bin | 1024 | d55db534043933c71af28e18e91f2db213fd359837b3cb9f29b0c3447fed841f |
| m1_mode0_perm_src_token_device.bin | 1024 | 97baae85b8a25ae55b2cbd4c6207fb37656a79cb8c16c5c9b779e69375daa192 |
| m1_mode0_res1_device.bin | 655360 | 28e9ed29b31dbaa635c4282799cbdcf2098b4f3e202c76285447b292a103743c |
| m1_mode0_res2_device.bin | 655360 | 55b1687cb8d958416b90c70934b3c61983566f057edb5707bc34a988d8605354 |
| m1_mode0_routed_output_device.bin | 327680 | 02416ebef56fe8ca9ec23c77ffd2a7dd334ca64a9acfbd4f5564ac0d7e57bc29 |
| m1_mode0_sgate_device.bin | 256 | 2caad3e52afa3462d300a70ce5733e581632e8dd075f9ab57be7223341033eeb |
| m1_mode0_shared_output_device.bin | 327680 | 1fd4f91cd4afcb6cb548a42aa3dc78b212a47a42fa4882de2ad9a256d83e22cb |
| m1_mode0_topk_weights_device.bin | 1024 | 455731bbfeaa13c38ae392d5217f93716c5172385be9c46b05c68dfac7151180 |
| m1_mode0_w_tk_packed_device.bin | 4096 | 04f3f57dd619d35d4cbfe6f2712e6facda79cd1df1228045aa99e892e06346a0 |
| m1_mode0_x_input_device.bin | 327680 | 6265e89f5564f533ea34847377d3004687f88b88cb40a7b43f46d8c436f325dd |
| m1_mode0_x_sorted_device.bin | 1310720 | e0558f380167c2f158aeb0d809ecb27d125be145f80835658ea20cd2d2d20817 |
| m1_mode0_xnorm_device.bin | 327680 | ccdb82aa427c064adfcc19b7fc567be7c4830a7aaeda3e14c51ae9835eb3b40e |
| m1_mode0_y_final_device.bin | 327680 | 358a6a94a6abe36ca73622aaa80633c1842c117895f6e93dda835a57a74fb1b4 |
| m1_mode0_y_shd_device.bin | 327680 | 67abfcd3db295fc97929851dda27168dfd4f361d617247fb99c16fe381b7c372 |
| m1_mode0_y_sorted_device.bin | 1310720 | 7ec3bcf2f8476ba8963b1c81d874a0120ca5c7cfa6069c2c1275535d96ab29aa |
| m1_mode1_a_qx_device.bin | 327680 | abe1dcd17ed651df3ae73c79cd6769c18465c29e5b9348d84c5fa599dc34c52e |
| m1_mode1_a_qx_shd_device.bin | 81920 | 47161be211ea04f44ca1f1bf819cd21e499527351cba68c4d282a0f6107036b7 |
| m1_mode1_a_scale_device.bin | 20480 | c38a530baa4ce225217851d46824006941c42bb143630064f702997113fe392b |
| m1_mode1_a_scale_shd_device.bin | 5120 | 454fc077760d9df513a7ac85c3e7a116c2dca163b1c5c7621d3b10ac06fd2089 |
| m1_mode1_expert_offsets_device.bin | 64 | b1b18910f7da486402c6c7b039638d76ec0873d24e11abe221d7d6899d9ed980 |
| m1_mode1_expert_token_counts_device.bin | 32 | 39169e9bced8a9aba2def574d5fc6b40fefe8213933435a585a5093d9fa6133e |
| m1_mode1_gamma1_input_device.bin | 5120 | d8752cad0f660495869b00205924c25e9819f37c23b34e5834432a05bdad5028 |
| m1_mode1_gamma2_input_device.bin | 5120 | 4ee21fee290fc03af05909ae9fc291894e30b9a577d5a9d60f94e36b6d865507 |
| m1_mode1_gu_device.bin | 655360 | f1de67994b5051ca8e29a75313002a0579e827241261b67fd3186e466e2e7bb5 |
| m1_mode1_gu_shd_device.bin | 163840 | 97a14b3254485f3cd28f6da37fb67e88184272f6f3f30e4241084870bb6c8985 |
| m1_mode1_h_qx_device.bin | 81920 | e0c603c259370e61fd5524e4ba856ab2ae5752908de51398e11666d5019ec70e |
| m1_mode1_h_qx_shd_device.bin | 20480 | f5c98f3b24c2679f74d5afa0c8a17eb93285bf42cba2b3101a0b25d9598412e2 |
| m1_mode1_h_scale_device.bin | 8192 | c3399818d238c1ba05e538805083b5aa17edd742abde4ad4cbae614d54f99d3f |
| m1_mode1_h_scale_shd_device.bin | 2048 | 95e9fc3da0146d9ec555da892363ad6d89d3110a31a50d751d3fc12a1693daf3 |
| m1_mode1_h_swiglu_device.bin | 327680 | a01fb3149bf1ce84e43274b1538ed95f8ff349aae5257e3021513d9af8ebe683 |
| m1_mode1_h_swiglu_shd_device.bin | 81920 | ee5c9cdf115343b27633b59bc45e85ed4353daa878a859504301c92877d33873 |
| m1_mode1_ids_device.bin | 1024 | d55db534043933c71af28e18e91f2db213fd359837b3cb9f29b0c3447fed841f |
| m1_mode1_inv_slot_device.bin | 1024 | f1c0ceb07394f25a70a24317f4846a3ae4731392ffae2af84aa1477b11a5c31e |
| m1_mode1_logits_device.bin | 1024 | 4b69fa06d10b07581ab7db05b2bf13eef9a51bc418731970fe584620b93b9ce3 |
| m1_mode1_moe_output_device.bin | 327680 | ed9de1e423b6c07cd07232c9ac3ed7b57d28416713ac13788a5c9f9fd0bee2a6 |
| m1_mode1_perm_expert_device.bin | 1024 | d55db534043933c71af28e18e91f2db213fd359837b3cb9f29b0c3447fed841f |
| m1_mode1_perm_src_token_device.bin | 1024 | 97baae85b8a25ae55b2cbd4c6207fb37656a79cb8c16c5c9b779e69375daa192 |
| m1_mode1_res1_device.bin | 655360 | 28e9ed29b31dbaa635c4282799cbdcf2098b4f3e202c76285447b292a103743c |
| m1_mode1_res2_device.bin | 655360 | 7bb1eec01efaf23b33e49b26eb8489931caa0bd027ef51eb443366e494a22b46 |
| m1_mode1_routed_output_device.bin | 327680 | 9560bcc162331888d152bebf4533f568c5c2c384dfd478828fb89207453f936c |
| m1_mode1_sgate_device.bin | 256 | e009e5c97dacf76d21e0dda8b23626b9ed4fd10377fe0235ccf32bc7a71d1b1a |
| m1_mode1_shared_output_device.bin | 327680 | 4520046653da94afc65e9d050a754358c143b3b17dd951ae6384c7c4ecc6b597 |
| m1_mode1_topk_weights_device.bin | 1024 | 2bde296efb0612fb54d43fc47d66bdc67bc0e4397e3c4c2cc3df86e238c002c6 |
| m1_mode1_w_tk_packed_device.bin | 4096 | b6e52a4f60c8a3f3c14bd8e960cd706689165fb150d1aa336a75a21059a39d95 |
| m1_mode1_x_input_device.bin | 327680 | 6265e89f5564f533ea34847377d3004687f88b88cb40a7b43f46d8c436f325dd |
| m1_mode1_x_sorted_device.bin | 1310720 | 4dd01e0312949548b82ce9a3e42bfd6e4cfb4839c77d317252e0b41b1629b129 |
| m1_mode1_xnorm_device.bin | 327680 | ccdb82aa427c064adfcc19b7fc567be7c4830a7aaeda3e14c51ae9835eb3b40e |
| m1_mode1_y_final_device.bin | 327680 | 4d59cb35b7f358c3adfc759783c88d27bab0f76ab2836b00602490ee52f505b6 |
| m1_mode1_y_shd_device.bin | 327680 | a89745550b944201cadfc278341068bc8718935186be636efcee813c9e66284f |
| m1_mode1_y_sorted_device.bin | 1310720 | b6ce082fbd04806d03dc25cec0bbc87eeab9e34cc3897014be10a9d815b3ac03 |
| m33_mode0_a_qx_device.bin | 327680 | 68bb25e17aeea7d24a5ff3f2aeb403b6acd685146d5198f71b6d90751ee9571d |
| m33_mode0_a_qx_shd_device.bin | 81920 | 458bb6243a12f6cb4d91e35c2e6dcc20eee9693cfed361f2f22f6a4aab9acf39 |
| m33_mode0_a_scale_device.bin | 20480 | aca5fab43bf45bae2d2c1ca3b2213212e9c107874dc96e843717b35ac05ed0ba |
| m33_mode0_a_scale_shd_device.bin | 5120 | cad0f60444e280514c27f07de8729042f2dde194c360b57334a25fb2c9331631 |
| m33_mode0_expert_offsets_device.bin | 64 | d5ab910400dc027d61e17758a9f082862322c8f29c43d7e3a82136835f8c2e6f |
| m33_mode0_expert_token_counts_device.bin | 32 | 18bedb0a8006823fae949da345985c91caabd160e7b4bb5b811cbe3487583472 |
| m33_mode0_gamma1_input_device.bin | 5120 | d8752cad0f660495869b00205924c25e9819f37c23b34e5834432a05bdad5028 |
| m33_mode0_gamma2_input_device.bin | 5120 | 4ee21fee290fc03af05909ae9fc291894e30b9a577d5a9d60f94e36b6d865507 |
| m33_mode0_gu_device.bin | 655360 | de3bbec2e43c843d298d91e61e7d91f6f0932b18ac31c64ecd512d2280619678 |
| m33_mode0_gu_shd_device.bin | 163840 | 783388f3f9925e2ba25546149983fb3a2fa43b65bd71350d592152819a238538 |
| m33_mode0_h_qx_device.bin | 81920 | 74e26db5362e07d28bd276a4e3c4b90d39f577920e3cf114014f721e15e5e47b |
| m33_mode0_h_qx_shd_device.bin | 20480 | 100f7521c1e6c329eb117f023d2055e959df0eaf6cac19c034507f749fa50dfc |
| m33_mode0_h_scale_device.bin | 8192 | cac334a7420b383acd410e5dc6e1ef3c0b8ec259c8078503f6522a4723b61b33 |
| m33_mode0_h_scale_shd_device.bin | 2048 | 935175550e8890d8c7674ab7ac46bfc25648fb9e8550bac6aea657af7be58fd9 |
| m33_mode0_h_swiglu_device.bin | 327680 | bfc2381654a8927c85417bfe0b9cd4f75d8778f3990a0449a2f5ada9204d37b9 |
| m33_mode0_h_swiglu_shd_device.bin | 81920 | cc3852727ddb08444cfe4464a5a4c0d9299ed3521f7b11d042fa6f9085120406 |
| m33_mode0_ids_device.bin | 1024 | ba1273e8370b33d5ecb3223d0d1f25c0fea20836d33c762a8e74ea4fbb4a747f |
| m33_mode0_inv_slot_device.bin | 1024 | 7647e52e67f27a57403a9be76bfbd757e1917855e2dadedc7671c82c9f97355a |
| m33_mode0_logits_device.bin | 1024 | 40678a3ae225fbb961c5ee9a40351f6bae7cddd4d589561fc80b2159166d0a8f |
| m33_mode0_moe_output_device.bin | 327680 | 40f20b231e5d509bd9505276307c609cc5d868b3f470750dfcbf5a7d69f1b914 |
| m33_mode0_perm_expert_device.bin | 1024 | a108dbe8a20fac3ef445a931f5a873f2bdf84387d4219e3045dd39cdcca1bf3f |
| m33_mode0_perm_src_token_device.bin | 1024 | 243d49e4a491e100a727afbf0fab306473af0a96928aa02885700630e1056e5a |
| m33_mode0_res1_device.bin | 655360 | 7109d8087e2851aa57ee12061989bf3c945f3367c58d8599e4bb1479c8f8d359 |
| m33_mode0_res2_device.bin | 655360 | 79e648562f9ec30e642e456336489af9bccc45e987f9155aab93b4bf700a1515 |
| m33_mode0_routed_output_device.bin | 327680 | 9f733db015c78516608af8ff2611c84fd4e47483df73d0492a7a45a28c56a614 |
| m33_mode0_sgate_device.bin | 256 | db162fb4730b21e9dea926acf28eca48daadedc86d5d42d2eb2b4dce3594f3e1 |
| m33_mode0_shared_output_device.bin | 327680 | 3cf7d72c533ea6765e87eac75ed7ee7b90fb64d24eb497379348fc2e87169217 |
| m33_mode0_topk_weights_device.bin | 1024 | 62be32bc9459064402fcab5d0e326d59f42e63ecd44ed7422b6fcd043d6e356b |
| m33_mode0_w_tk_packed_device.bin | 4096 | 356780f448828e90889c67b075c7a888e81e6e68a0ad222e191821b66de1732a |
| m33_mode0_x_input_device.bin | 327680 | 4798afd1ae8869b9eeecd6ec9311a4212688d1d13c41c72bc39534d6666e17cc |
| m33_mode0_x_sorted_device.bin | 1310720 | 421e11afc1c2e4f34d63512ae43f00112938e1bd44d579127cff05c8856f5eeb |
| m33_mode0_xnorm_device.bin | 327680 | 568d41f38cd1dd3b6eb2b354d26f3a444e27b8170dbacd58354b370546291e4f |
| m33_mode0_y_final_device.bin | 327680 | 0d165bb2610d82ca3f61ddb22f70e6888003dfca34026cda57c2992057438195 |
| m33_mode0_y_shd_device.bin | 327680 | ffb68b71e840ec1686e7d3d2707dc9c1daa512b61e8fee2f095fc9b377a3e21d |
| m33_mode0_y_sorted_device.bin | 1310720 | 53fec451634070eb67f1586efb94c8e25167e785b8ef47aa698a42474ab061cf |
| m33_mode1_a_qx_device.bin | 327680 | 3b4f37b66e2dcc74f1acf8f32185dfdc602216368447f0c08323489eabf694f6 |
| m33_mode1_a_qx_shd_device.bin | 81920 | 44e5360949a4942f9d1330403ce27055cb7504da57c17ddd196df4e9a41fbced |
| m33_mode1_a_scale_device.bin | 20480 | e7cf4ffd672d9d0322178de9caea172e38e56b3094fd5ad4d6d0b1fba8276d9e |
| m33_mode1_a_scale_shd_device.bin | 5120 | b239217deab8ee895f19effa32cd3ec5d3ac139daf4a813f91a0526f2ef77675 |
| m33_mode1_expert_offsets_device.bin | 64 | fcfb283ac7ef402f58a513224468fc46481f78f96c44a071f0a6f69c2f4a1ce9 |
| m33_mode1_expert_token_counts_device.bin | 32 | 66290f4e1b83760ce2db2e5108d69c4f01feac79082ff37222869ec669fd4990 |
| m33_mode1_gamma1_input_device.bin | 5120 | d8752cad0f660495869b00205924c25e9819f37c23b34e5834432a05bdad5028 |
| m33_mode1_gamma2_input_device.bin | 5120 | 4ee21fee290fc03af05909ae9fc291894e30b9a577d5a9d60f94e36b6d865507 |
| m33_mode1_gu_device.bin | 655360 | 3f55a0d04f7da3fcc2f6081efb9ea02e9931e5073d1df3bff75e884ddcf07a27 |
| m33_mode1_gu_shd_device.bin | 163840 | 1150abb0ba35ef9585af76a39b647a5b4bf4ed78b32e3576cc486b3a48286b24 |
| m33_mode1_h_qx_device.bin | 81920 | ad26154a9052a7f103583edd0b55e14137110a0597350d11f0848201b26217db |
| m33_mode1_h_qx_shd_device.bin | 20480 | c2e8b304656fe15a4d0426d9c1df81aad83293839e34e3428e584543e50cd27c |
| m33_mode1_h_scale_device.bin | 8192 | a0c6897f0d0c173bdcadbd4b2ba15056f9de0b170b456b460ff501e553707259 |
| m33_mode1_h_scale_shd_device.bin | 2048 | e16afae898772edff5f78f67a677db1c6db9661de417035d16af1156139505d5 |
| m33_mode1_h_swiglu_device.bin | 327680 | b48e0fba059ce4df325465e21c166a50d3775127b5adf58fba61bf1b3719e31d |
| m33_mode1_h_swiglu_shd_device.bin | 81920 | 64559edc31c4254dd0c753ed9e49f567de926409af087708b98ad583cff0deae |
| m33_mode1_ids_device.bin | 1024 | ee382a5d5310ed634358b073b923f37f179c8ca25addc90dc43673f62b8faab6 |
| m33_mode1_inv_slot_device.bin | 1024 | cfc2dd65c3720ba9b487c2a75d4125ff1493178f0be85c4e19c416e1be825829 |
| m33_mode1_logits_device.bin | 1024 | c9286db790ac292e805945b9ee0a8ae8c60aa7e90b1201bb7b6c14891b81d0f0 |
| m33_mode1_moe_output_device.bin | 327680 | a566bc7036ec3cdf11b647f88f31d459b6d86833c42761690542858818a7f3ff |
| m33_mode1_perm_expert_device.bin | 1024 | 51061ada38bf634732f570aecaee56ef0d2d95fdd3097c661483160a94f2e762 |
| m33_mode1_perm_src_token_device.bin | 1024 | 007a7a2d1798a178941f6e4b177003d09d1c5f03eea14c17c57e078af9a469ac |
| m33_mode1_res1_device.bin | 655360 | 7109d8087e2851aa57ee12061989bf3c945f3367c58d8599e4bb1479c8f8d359 |
| m33_mode1_res2_device.bin | 655360 | 02aece80e8b1afd78cc4d3da43cadc1f0d0c232dcf4d95a4d1412e4cbc350c62 |
| m33_mode1_routed_output_device.bin | 327680 | f3db606f8a2f67441b9427da6bfe92bc240370e92e047ce87c75a842096276c4 |
| m33_mode1_sgate_device.bin | 256 | 0ae8cb86afee0c072575dc3b10cd3ee320ea743411589aed138c448e2e226419 |
| m33_mode1_shared_output_device.bin | 327680 | bdd973b3285f0aa73a401ed3ed256e214568dea83d17974197db5ea07f645a22 |
| m33_mode1_topk_weights_device.bin | 1024 | 6cbf1410ed9b5fdd3aa045c32beaa6fb663c62ad300b534c16287a8217d2c1d4 |
| m33_mode1_w_tk_packed_device.bin | 4096 | a83fcdb13e8c1fd4bcb2e446a4340ad2a6733b9e88284f937d44a058e0697a40 |
| m33_mode1_x_input_device.bin | 327680 | 4798afd1ae8869b9eeecd6ec9311a4212688d1d13c41c72bc39534d6666e17cc |
| m33_mode1_x_sorted_device.bin | 1310720 | 42fb1ce9252158b9622e934095e2ae989626bc59afd819e2b05a0564efbf206d |
| m33_mode1_xnorm_device.bin | 327680 | 568d41f38cd1dd3b6eb2b354d26f3a444e27b8170dbacd58354b370546291e4f |
| m33_mode1_y_final_device.bin | 327680 | 6cc6f4fcdff3bae04170e62ad5db73242e2adb67ae6449e89275a26eb9d9200f |
| m33_mode1_y_shd_device.bin | 327680 | d1a532c25f86b3922add2fe61a9acabc419cc1451918ffc9fc15a757178edf84 |
| m33_mode1_y_sorted_device.bin | 1310720 | 0f6df5b6397f543886d114ffa822502cb2ba6a726246a0ea4e890c53bbddd208 |
