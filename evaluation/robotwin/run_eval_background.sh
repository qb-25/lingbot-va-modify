#!/bin/bash
# 后台运行 Server + Client 的便捷脚本
# 用法: bash evaluation/robotwin/run_eval_background.sh [save_root] [task_name]
# 示例: bash evaluation/robotwin/run_eval_background.sh results/ adjust_bottle

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

save_root="${1:-results/}"
task_name="${2:-adjust_bottle}"
PORT=29056
LOG_DIR="${LOG_DIR:-./eval_logs}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
CLIENT_LOG="$LOG_DIR/client_$(date +%Y%m%d_%H%M%S).log"

echo "========== 1. 启动 Server（后台）=========="
nohup bash evaluation/robotwin/launch_server.sh >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID, 日志: $SERVER_LOG"

echo "========== 2. 等待端口 $PORT 就绪 =========="
for i in {1..60}; do
  if (echo >/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    echo "端口 $PORT 已就绪."
    break
  fi
  if [ $i -eq 60 ]; then
    echo "超时: 端口 $PORT 未就绪，请检查 Server 日志: $SERVER_LOG"
    exit 1
  fi
  sleep 2
done

echo "========== 3. 启动 Client（后台）=========="
# Client 会 chdir 到 RoboTwin，save_root 相对于 RoboTwin 工作目录
nohup env task_name="$task_name" save_root="$save_root" bash -c '
  task_name="${task_name:-adjust_bottle}"
  save_root="${save_root:-results/}"
  bash evaluation/robotwin/launch_client.sh "$save_root" "$task_name"
' >> "$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!
echo "Client PID: $CLIENT_PID, 日志: $CLIENT_LOG"

echo ""
echo "========== 后台任务已启动 =========="
echo "  Server PID: $SERVER_PID  日志: $SERVER_LOG"
echo "  Client PID: $CLIENT_PID  日志: $CLIENT_LOG"
echo "  查看日志: tail -f $CLIENT_LOG  或  tail -f $SERVER_LOG"
echo "  评估结果: 见下方「结果保存位置」"
echo ""
