from __future__ import annotations

import argparse
import json
import secrets

from .auth import token_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="LIBERO gateway administration helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    token_parser = subparsers.add_parser("token", help="generate an agent bearer token")
    token_parser.add_argument("agent_id")
    args = parser.parse_args()

    if args.command == "token":
        token = "lbr_" + secrets.token_urlsafe(32)
        print("Give this token to the agent once; it is not stored by this command:")
        print(token)
        print("Add this entry to agents.json:")
        print(
            json.dumps(
                {
                    args.agent_id: {
                        "token_sha256": token_sha256(token),
                        "allowed_benchmarks": [
                            "libero_spatial",
                            "libero_object",
                            "libero_goal",
                            "libero_10",
                        ],
                        "max_sessions": 1,
                    }
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
