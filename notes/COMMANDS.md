在当前 Pod 跑单回合task可用：
```
  cd /root/Efficient-WAM
  PATH="$PWD/.venv/bin:$PATH" \
  VK_ICD_FILENAMES=/tmp/efficientwam-nvidia-egl-icd.json \
  EFFICIENT_WAM_LOG_ROOT=/workspace/results/efficientwam \
  bash inference/robotwin/EfficientWAM/eval.sh \
    --config inference/robotwin/EfficientWAM/deploy_policy.local.yml \
    --task adjust_bottle --episode-num 1
```