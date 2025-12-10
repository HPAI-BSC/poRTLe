# OpenCode Agent

A poRTLe agent that uses [opencode-ai](https://github.com/opencode-ai/opencode) for RTL code generation.

## Quick Start

1. **Configure API Key**

   Copy the example auth file and add your OpenRouter API key:

   ```bash
   cp auth.example.json auth.json
   ```

   Edit `auth.json` and replace `EMPTY` with your OpenRouter API key:

   ```json
   {
     "openrouter": {
       "type": "api",
       "key": "sk-or-v1-your-api-key-here"
     }
   }
   ```

   Get an API key at [openrouter.ai](https://openrouter.ai/) or use any other model provider supported by opencode.

## Troubleshooting

If poRTLe fails to build the opencode agent, you can manually build it for debugging:

```bash
# First, ensure the base image is built
cd ../../docker && ./build.sh

# Then build the agent
./build_agent.sh
```

## Notes

- The `auth.json` file is copied into the container at build time
- All OpenCode state and logs persist in `/code/rundir/.opencode/` after container exit
