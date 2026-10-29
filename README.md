# Hardware Introspection Agent

An AI-powered Linux hardware inspection assistant built with Python and tool calling.

The Hardware Introspection Agent allows users to inspect their computer's hardware through natural-language questions. Instead of manually running multiple Linux commands, users can ask questions such as “How much RAM do I have?” or “What kind of CPU is installed?” and receive readable answers based on system information collected by predefined hardware inspection tools.

Developed as part of an Operating Systems and Agentic Shell college project.

## Overview

The agent combines a conversational language model with specialized, read-only hardware inspection tools. It interprets user questions, selects relevant tools, collects hardware information from the Linux operating system, and presents the results in natural language.

## Features

- Natural-language interaction with the hardware assistant.
- Hardware inspection for CPU, RAM, GPU, storage, USB, PCI devices, motherboard, and BIOS.
- AI-powered selection of relevant inspection tools.
- Linux-focused hardware information collection.
- Read-only inspection design.
- Modular architecture and error handling.
- Automated tests for agent and hardware-tool behavior.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core implementation |
| Linux | Target operating system |
| OpenAI or Google Gemini API | Conversational reasoning, depending on implementation |
| Python dotenv | Environment configuration |
| Pytest | Automated testing |
| Linux hardware utilities | Hardware information collection |

The available AI provider and model depend on the implementation and repository configuration.

## Project Structure

```text
operating-Systems-project/
├── hardware_agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── system_utils.py
│   └── tools.py
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   └── test_tools.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

- Ubuntu or another compatible Linux distribution.
- Python 3.10 or newer.
- pip and Python virtual environment support.
- A valid API key for a supported language model provider.
- Required Linux hardware inspection utilities.

On Ubuntu, install the common system utilities:

```bash
sudo apt update
sudo apt install -y dmidecode lshw pciutils usbutils util-linux procps
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yashshete55/OS-PROJECT.git
cd OS-PROJECT
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local environment file:

```bash
cp .env.example .env
```

Configure the provider and API key according to the implementation. Never commit `.env` or publish real API keys.

## Usage

Start the agent:

```bash
python main.py
```

Example questions:

- How much RAM do I have?
- What kind of CPU is installed?
- Show my storage devices.
- What USB devices are connected?
- Tell me about my GPU.

The exact output depends on the hardware, Linux distribution, available utilities, permissions, and configured language model.

## Testing

Run the automated tests:

```bash
pytest -v
```

## Architecture

```text
User Question
     |
     v
Interactive Interface
     |
     v
Conversational AI Agent
     |
     v
Hardware Tool Selection
     |
     v
Linux Inspection Utilities
     |
     v
Hardware Information
     |
     v
Natural-Language Response
```

## Security and Limitations

- The project is intended for hardware inspection rather than system configuration changes.
- Some hardware details may require additional permissions.
- API access requires valid credentials and network connectivity.
- Hardware information may be incomplete if utilities are unavailable.
- Unavailable information should be reported as unknown rather than fabricated.
- Keep API keys outside version control.

## Future Enhancements

- Additional hardware diagnostic tools.
- Improved conversational context and follow-up questions.
- Hardware reports in JSON or text format.
- Better error reporting and provider configuration.
- Cross-distribution compatibility testing.
- Integration with a larger Agentic Shell system.

## Academic Project

**Project:** Hardware Introspection Agent  
**Domain:** Operating Systems, Python, and Artificial Intelligence  
**Primary objective:** Make Linux hardware information accessible through a natural-language interface.

**License:** Specify the applicable license if one is included in the repository.
