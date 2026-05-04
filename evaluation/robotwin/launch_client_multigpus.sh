#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=eval_run_paths.inc.sh
source "$SCRIPT_DIR/eval_run_paths.inc.sh"
cd "$REPO_ROOT" || exit 1

# $1: save_root — 默认 auto：从 CKPT_PATH + 时间戳 推导，或与 server 写入的 logs/.eval_run.env 对齐
# $2: task_list_id (默认 0)
save_root_arg="${1:-auto}"
task_list_id="${2:-0}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/data/qianbin/RoboTwin}"
EVAL_RUN_ENV_FILE="${EVAL_RUN_ENV_FILE:-$LINGBOT_EVAL_RUN_ENV_DEFAULT}"

if [[ "$save_root_arg" == "auto" || "$save_root_arg" == "AUTO" ]]; then
  if [[ -f "$EVAL_RUN_ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$EVAL_RUN_ENV_FILE"
  fi
  if [[ -n "${EVAL_ROBOTWIN_SAVE_ROOT:-}" ]]; then
    save_root="$EVAL_ROBOTWIN_SAVE_ROOT"
  else
    EVAL_CKPT_SLUG="$(lingbot_eval_ckpt_slug "${CKPT_PATH:-}")"
    if [[ -z "$EVAL_CKPT_SLUG" ]]; then
      echo "[launch_client_multigpus] ERROR: auto save_root needs CKPT_PATH or ${EVAL_RUN_ENV_FILE} from server." >&2
      exit 1
    fi
    ts="${EVAL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    save_root="${ROBOTWIN_ROOT%/}/eval_va/${EVAL_CKPT_SLUG}__${ts}"
  fi
else
  save_root="$save_root_arg"
fi

mkdir -p "$save_root"
echo "[launch_client_multigpus] save_root=$save_root  task_list_id=$task_list_id"

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${3:-0}
test_num=${4:-100}
start_port=29556 
num_gpus=8

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
  echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

read -r -a task_names <<< "${task_groups[$task_list_id]}"

echo "task_list_id=$task_list_id"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"

log_dir="./logs"
mkdir -p "$log_dir"

echo -e "\033[32mLaunching ${#task_names[@]} tasks. GPUs assigned by mod ${num_gpus}, ports starting from ${start_port} incrementing.\033[0m"

pid_file="pids.txt"
> "$pid_file"

batch_time=$(date +%Y%m%d_%H%M%S)

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    gpu_id=$(( i % num_gpus ))
    port=$(( start_port + i ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${log_dir}/${task_name}_${batch_time}.log"

    echo -e "\033[33m[Task $i] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --save_root "${save_root}" \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num ${test_num} \
        --port ${port} > "$log_file" 2>&1 &
    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

echo -e "\033[32mAll tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"
# 默认等待全部 client 结束，避免 DLC「启动命令」主进程退出后子进程被一起杀掉。
# 若只想后台拉起不等待: export EVAL_CLIENT_WAIT_ALL=0
if [ "${EVAL_CLIENT_WAIT_ALL:-1}" = "1" ]; then
    echo "Waiting for all evaluation clients (EVAL_CLIENT_WAIT_ALL=1)..."
    wait
fi
