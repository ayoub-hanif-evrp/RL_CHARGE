# Hybrid PPO TRAIN/VAL pilots

TEST is never used here.

| Archive | Meaning |
| --- | --- |
| `pre_action_fix/` | Failed first pilot (canonical copy). |
| `post_action_reward_fix/` | Second pilot after station-revisit mask, `L_remaining` failure term, and parent-balanced VAL selection. |

Do not keep duplicate `seed_*` trees beside these archives.
