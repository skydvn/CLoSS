#!/usr/bin/env bash
# NoRGa as a domain-incremental baseline, on exactly the data setup MoSS uses: the MoSS
# config supplies dataset/domains/seed, and every key NoRGa reads is overridden here, so
# MoSS-specific values cannot leak into NoRGa training.
# Requires utils/norga_data.py::dil_task_dataset to match utils/dil_data_manager.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=${BASE:-exps/moss/moss_domainnet_dil.json}
python3 main.py --config "$BASE" --set \
  model_name=norga prefix=norga_dil scenario=dil \
  backbone_type=vit_base_patch16_224_in21k pretrained=true \
  norga=true gate_act=tanh 'act_scale_init=[1.0,1.0]' \
  'prompt_layers=[0,1,2,3,4]' prompt_length=20 prompt_init=uniform \
  prompt_init_from_prev=true prompt_momentum=0.01 \
  wtp_epochs=20 wtp_batch_size=24 wtp_optimizer=adam wtp_lr=0.005 \
  wtp_weight_decay=0.0 wtp_scheduler=cosine train_mask=false reg=0.001 cr_temperature=0.8 \
  stats_cov_type=diag stats_shrink=0.0001 \
  tii_head=linear tii_n_centroids=1 tii_ca_epochs=30 tii_ca_lr=0.005 \
  ca_epochs=30 ca_lr=0.005 ca_samples_per_class=32 ca_weight_decay=0.0005 \
  num_workers=8 eval_batch_size=128 eval_oracle=true \
  "$@"
