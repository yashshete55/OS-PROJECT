# Hardware Introspection Agent

**An AI-powered Linux hardware inspection assistant built with Python and tool calling.**

The Hardware Introspection Agent allows users to inspect their computer's hardware through natural-language questions. Instead of manually running multiple Linux commands, users can ask questions such as *“How much RAM do I have?”* or *“What kind of CPU is installed?”* and receive readable answers based on system information collected by predefined hardware inspection tools.

Developed as part of an Operating Systems and Agentic Shell college project.

---

## Overview

The agent combines a conversational language model with specialized, read-only hardware inspection tools. It interprets user questions, selects relevant tools, collects hardware information from the Linux operating system, and presents the results in natural language.

The agent is designed to report actual system information rather than inventing hardware specifications.

## Features

- **Natural-language interaction** — Ask questions about your computer in plain English.
- **Hardware inspection** — Retrieve information about CPU, RAM, GPU, storage, USB devices, PCI devices, motherboard, and BIOS.
- **AI-powered tool selection** — Uses a supported language model to identify which inspection tools are relevant to a question.
- **Linux-focused design** — Uses Linux system interfaces and hardware inspection utilities.
- **Read-only operation** — Designed to inspect hardware without intentionally modifying system configuration.
- **Modular architecture** — Separates conversation logic, configuration, hardware utilities, and tool definitions.
- **Error handling** — Provides diagnostic messages for missing configuration and failed inspections.
- **Automated tests** — Includes tests for agent behavior and hardware tools.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core implementation |
| Linux | Target operating system |
| OpenAI or Google Gemini API | Conversational reasoning and tool selection, depending on configured provider support |
| Python dotenv | Environment configuration |
| Pytest | Automated testing |
| Linux hardware utilities | Hardware information collection |

*The available AI provider and model depend on the implementation and configuration in the repository.*

## Project Structure

```text
operating Systems project/
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

### Module Responsibilities

- `main.py` — Application entry point and interactive interface.
- `hardware_agent/agent.py` — Conversational agent and tool-calling workflow.
- `hardware_agent/config.py` — Environment variables and provider configuration.
- `hardware_agent/system_utils.py` — System information and hardware inspection utilities.
- `hardware_agent/tools.py` — Hardware inspection tools exposed to the agent.
- `tests/` — Automated tests for agent and tool behavior.
- `.env.example` — Example environment configuration without real credentials.
- `requirements.txt` — Python dependency list.

## Prerequisites

Before running the project, ensure that you have:

- Ubuntu or another compatible Linux distribution.
- Python 3.10 or newer.
- `pip` and Python virtual environment support.
- An API key for a language model provider supported by your implementation.
- Required Linux hardware inspection utilities.

Install the common system utilities on Ubuntu:

```bash
sudo apt update
sudo apt install -y dmidecode lshw pciutils usbutils util-linux procps
```

Some hardware information, particularly motherboard and BIOS details, may require elevated permissions. Run the application with the minimum permissions needed.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nagarjuna5362/operatoing-Systems-project.git
cd operatoing-Systems-project
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

Create your local `.env` file from the example:

```bash
cp .env.example .env
```

Open `.env` and configure the provider and API key according to the options supported by your implementation.

Example configuration for Gemini:

```dotenv
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
COMMAND_TIMEOUT=10
MASK_SERIALS=true
```

If using OpenAI, configure the corresponding OpenAI variables instead.

**Important:** Never commit `.env` or publish real API keys. Keep credentials in your local environment.

## Usage

Start the agent:

```bash
python main.py
```

Ask questions interactively, for example:

```text
Hardware Introspection Agent > How much RAM do I have?

Hardware Introspection Agent > What kind of CPU is installed?

Hardware Introspection Agent > Show my storage devices.

Hardware Introspection Agent > What USB devices are connected?

Hardware Introspection Agent > Tell me about my GPU.
```

The agent should select the relevant inspection tool, retrieve available system information, and return a natural-language answer.

*The exact output depends on the hardware, Linux distribution, available utilities, permissions, and configured language model.*

## Available Hardware Information

The project is designed to inspect categories including:

| Category | Example question |
|---|---|
| CPU | What processor is installed? |
| Memory | How much RAM is available? |
| GPU | Which graphics devices are detected? |
| Storage | What disks and storage devices are present? |
| USB | Which USB devices are connected? |
| PCI | What PCI devices are installed? |
| Motherboard | What motherboard information is available? |
| BIOS | What BIOS information can be retrieved? |

Actual coverage depends on the inspection tools implemented in the repository and the information exposed by the system.

## Testing

Run the automated test suite:

```bash
pytest -v
```

The tests help verify agent behavior and hardware-tool functionality. Passing automated tests does not necessarily verify live API access or every hardware configuration.

## Architecture

The high-level workflow is:

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

The language model interprets the question and selects an appropriate predefined tool. The tool retrieves system information, which the agent uses to construct the response.

## Security and Limitations

- The project is intended for hardware inspection, not system administration or configuration changes.
- Hardware information may be incomplete when permissions or required utilities are unavailable.
- Some commands may behave differently across Linux distributions.
- API access requires valid provider credentials and network connectivity.
- API keys must remain outside version control.
- Do not assume every answer is available on every computer; unavailable information should be reported as unknown or unsupported rather than fabricated.

## Future Enhancements

Potential improvements include:

- Additional hardware diagnostic tools.
- More robust conversational context and follow-up questions.
- Structured hardware reports in JSON or text format.
- Improved error reporting and provider configuration.
- Cross-distribution compatibility testing.
- Integration with a larger Agentic Shell system.

## Academic Project

**Project:** Hardware Introspection Agent  
**Domain:** Operating Systems, Python, and Artificial Intelligence  
**Primary objective:** Make Linux hardware information accessible through a natural-language interface.

---

**License:** MIT, if the repository includes an MIT license file.
