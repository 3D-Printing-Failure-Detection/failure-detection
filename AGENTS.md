# Long-running work

- Start long training and evaluation commands in a named tmux session, with logs
  on the data volume, so the user can disconnect from SSH and reconnect later.
- Give the user the session name and attach command. Preserve the exit status
  and keep completed logs available for review.
- Do not restart or discard an in-flight experiment just to move it into tmux.
  Check whether it is already detached from SSH; if so, monitor it from tmux and
  use tmux for subsequent launches.
- For an interactive Codex/agent session, recommend launching the agent inside
  tmux. Do not start a duplicate agent for an already-active task.

The parent workspace's storage quota and artifact-location rules still apply.
