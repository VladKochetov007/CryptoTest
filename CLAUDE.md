# Baseline Project Guidelines

## MCP Tools

### Context7 - MUST USE
**When to use**:
- Always when looking up library documentation
- When troubleshooting library-specific issues
- Before implementing features with external dependencies

**How to use**:
1. `mcp__context7__resolve-library-id` - Find the library ID
2. `mcp__context7__query-docs` - Query documentation with the library ID

**Why**: Ensures current, accurate documentation rather than outdated knowledge.

### DeepWiki - USE FOR CODEBASE INTELLIGENCE
**When to use**:
- Understanding project structure and architecture
- Getting context about codebases (this repo or dependencies)
- Asking questions about how specific features work
- Exploring undocumented or complex code flows

**How to use**:
1. `mcp__deepwiki__read_wiki_structure` - Get documentation topics for a repo
2. `mcp__deepwiki__read_wiki_contents` - View full documentation
3. `mcp__deepwiki__ask_question` - Ask specific questions about the repo

**Why**: Fast AI-powered navigation of code context without manual exploration.

### DevRag - USE FOR SUBTITLE SEARCH
**When to use**:
- Before reading subtitle files directly
- When looking for mentions of an experiment, setup, material, voltage, thrust, efficiency, ozone, collector geometry, or ion wind
- When the answer is likely buried in the `subtitles/` transcripts

**Project setup**:
- MCP config: `.mcp.json`
- Active DevRag config: `config.json`
- Binary: `.tools/devrag/bin/devrag`
- Vector DB: `.devrag/vectors.db`
- Models: `.devrag/models/`
- Indexed documents: `subtitles/episode-01.md` through `episode-17.md` in russian.

**How to use**:
1. Prefer the DevRag MCP server for semantic lookup instead of opening multiple transcript files.
2. Start with natural-language search queries such as `серебряное покрытие`, `тяга`, `КПД`, `напряжение`, or `коллектор` and `эмитер`.
3. Read only the returned chunks first. Open the source subtitle file only if more local context is needed.
4. If subtitle files are added or changed, reindex or restart the DevRag server before relying on new results.

**CLI fallback**:
- Search: `./.tools/devrag/bin/devrag --config config.json search --top-k 5 --output text "ионный ветер"`
- List indexed docs: `./.tools/devrag/bin/devrag --config config.json list --output text`
- Reindex one file: `./.tools/devrag/bin/devrag --config config.json reindex ./subtitles/episode-01.md`

**Important details**:
- DevRag indexes `.md` files, so subtitles were renamed from `.txt` to `.md`.
- Flags must come before the search query in the CLI.
- The current config uses `chunk_size: 1200` and `search_top_k: 10` because transcripts need more context than normal docs.
- Use DevRag as retrieval, not as ground truth. Quote or summarize only after checking the returned chunk text.

## Dev Experts Agents - USE PROACTIVELY

**Always use these agents when the task matches:**

| Agent | Trigger |
|-------|---------|
| `dev-experts:reviewer` | Before ANY commit/PR, or when reviewing code changes |
| `dev-experts:tester` | When writing or updating tests |
| `dev-experts:architect` | When planning new features or significant changes |
| `dev-experts:python-dev` | When writing/reviewing Python code |
| `dev-experts:rust-dev` | When writing/reviewing Rust code |
| `dev-experts:cpp-dev` | When writing/reviewing C++ code |
| `dev-experts:devops` | When debugging production/infrastructure issues |

**How**: Use the Task tool with `subagent_type="dev-experts:<agent-name>"`

## Skills - USE WHEN APPLICABLE

| Skill | When to Use | How to Invoke |
|-------|-------------|---------------|
| `datetime` | Need current date/time for logging, file naming, timestamps, APIs | `/datetime` or use `date` commands from skill |
| `tmux` | Interactive programs: REPLs, debuggers, dev servers, parallel tasks | `/tmux` - use `tmux-ctl` commands |
| `anti-ai-slop` | After and before writing code, before commit - clean up AI-generated bloat and prevent new one | `/anti-ai-slop` |
| `arxiv-search` | Research papers: finding relevant work, state-of-the-art, citations, physics/ML/systems | `/arxiv-search` - call proactively when exploring topics |
| `97-dev` | Timeless programming wisdom for clean, maintainable code patterns | `/97-dev` - call when refactoring or designing |
| `prompt-improver` | Auto-triggered on vague prompts to ask clarifying questions | Automatic via hook |

### Project Skills (`.claude/skills/`)

| Skill | When to Use |
|-------|-------------|
| `ml-paper-writing` | Any paper writing task: drafting sections, structuring arguments, citations. Templates at `.claude/skills/ml-paper-writing/templates/icml2026/`. Use proactively without being asked. |
| `academic-plotting` | Any figure request. Data plots (loss, bar, ablation) via matplotlib; architecture/pipeline diagrams via Gemini. |
| `presenting-conference-talks` | Converting the paper to slides: generates Beamer PDF + PPTX with speaker notes. |
| `systems-paper-writing` | OSDI/SOSP/ASPLOS/NSDI only. Not relevant for ICML26. |

### Trading Skills (`.claude/skills/` — pump.fun task)

Sourced from `agiprolabs/claude-trading-skills`. Load with `Read .claude/skills/<name>/SKILL.md` before implementing anything in that domain.

| Skill | When to Use |
|-------|-------------|
| `pumpfun-mechanics` | Bonding curve math, graduation thresholds, CreateEvent/TradeEvent/CompleteEvent parsing — any pump.fun on-chain logic |
| `market-microstructure` | Order flow analysis, buy/sell imbalance, volume classification, momentum scoring — exit signal construction |
| `exit-strategies` | Tiered take-profit, trailing stops, time-based and leader-sell exits — Part 2 exit model |
| `feature-engineering` | Feature construction from price/volume/on-chain data with no-lookahead guarantees — pre-buy scoring features |
| `signal-classification` | XGBoost/LightGBM classifiers with walk-forward CV and SHAP importance — ML scoring model |
| `walk-forward-validation` | Purged CV, embargo periods, deflated Sharpe ratio — backtest validation for exit strategies |
| `token-holder-analysis` | Gini, HHI, top-N concentration, insider detection — holder distribution features |
| `sybil-detection` | Co-trade clustering, wash trade detection, bundler identification — cleaning data and scoring quality |
| `wallet-profiling` | Win rate, trading style classification, PnL tracking — deployer wallet behavioral scoring |
| `vectorbt` | Vectorized backtest with Sharpe/Sortino/Calmar, parameter sweeps — exit strategy simulation |
| `yellowstone-grpc` | Geyser gRPC streaming, subscription filters — architecture reference for 200-300ms pre-buy pipeline |
| `strategy-framework` | Standardized entry/exit/sizing/risk template — use before implementing any strategy |

### AI Research Skills Library (`.claude/skills/ai-research-skills/`)

85 skills for the full research lifecycle. Relevant categories:

| Category | Path |
|----------|------|
| Post-Training / MARL / RL | `06-post-training/` |
| Agents | `14-agents/` |
| Evaluation | `11-evaluation/` |
| Distributed Training | `08-distributed-training/` |
| Research Ideation | `21-research-ideation/` |
| ML Paper Writing | `20-ml-paper-writing/` |

### Quick Skill Reference

**datetime**: `date +%Y-%m-%d`, `date +%s` (epoch), `date -u +"%Y-%m-%dT%H:%M:%SZ"` (ISO UTC)

**tmux**:
- `tmux-ctl eval "command"` - one-off execution
- `tmux-ctl repl python "expr"` - REPL one-shot
- `tmux-ctl start "npm run dev" --name=server` - background process
- `tmux-ctl logs server` - view output

**anti-ai-slop**: Removes unnecessary comments, redundant checks, single-use variables, style drift from branch diff

## Code Quality

### Comments
**Only write comments for**:
- Complex algorithms that aren't immediately obvious
- Non-obvious workarounds for known issues/bugs
- Critical "why" explanations that code cannot express

**99% of code should be self-explanatory through**:
- Clear, descriptive naming
- Proper structure and organization
- Following language best practices

## Git Workflow

### Commit Messages
Use **Conventional Commits** format:
- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test updates
- `chore:` - Maintenance tasks

Example: `feat: add user authentication system`

### Branch Naming
Use **type prefixes**:
- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring

Example: `feature/user-authentication`

## Engineering Philosophy

### Approach
- **Pragmatic over perfect** - Ship working solutions, iterate if needed
- **Question assumptions** - Always ask clarifying questions before building
- **YAGNI** - Don't over-engineer; build what's needed now
- **KISS** - Prefer simple solutions over complex ones

### Communication Style
- **Ask questions** when requirements are unclear or ambiguous
- **Brainstorm ideas** - discuss different approaches
- **Find optimal approach** - balance simplicity, maintainability, and effectiveness
- **Avoid over-engineering** - question if complexity is truly necessary
- **Be concise** - get to the point, focus on solutions

## Testing
- Tests will be specified when needed
- Don't assume - ask if testing is required

---

# Code quality guidelines

```python
# Bad Code
def foo(a):
  q = len(a)
    for i in range(len(a)):
	r = a[i]
	b = a[a[i]] % q
	a[i] = q*b + r
  for i in range(len(a)):
    a[i] = a[i] // q
  return a

# Good Code
def transform_list(input_list, transformer):
    """
    Transforms a list using a provided transformer function.
    
    - `input_list`: list of values to transform, shape: (n,)
    - `transformer`: function to process each value
    
    Returns: list of transformed values, shape: (n,)
    """
    transformed_list = []
    for value in input_list:
        transformed_value = transformer(value)
        transformed_list.append(transformed_value)
    return transformed_list


def transform_value(value):
    """
    Default transformation function.
    
    - `value`: input value
    
    Returns: transformed value based on length calculation
    """
    length = len(value)
    transformed_value = value % length
    return length * transformed_value + value
```

```python
# Bad Code
def make_pizza():
    # Mix the ingredients for the dough
    flour = 2  # cups
    water = 1  # cup
    yeast = 0.25  # teaspoon
    salt = 1  # teaspoon
    dough = flour + water + yeast + salt

    # Knead the dough
    for i in range(10):
        dough *= 2  # Knead the dough by doubling its volume

    # Roll out the dough
    dough_thickness = 0.25  # inch
    dough = dough / dough_thickness

    # Apply the tomato sauce
    sauce = 0.5  # cups
    pizza = dough + sauce  # Add sauce to the dough

    # Add the cheese
    cheese = 1.5  # cups
    pizza += cheese  # Add cheese to the pizza

    # Preheat the oven
    oven_temperature = 450  # Fahrenheit

    # Set the timer for baking
    timer = 20  # 20 minutes

    # Bake the pizza in the oven
    baking_time = 0
    while baking_time < timer:
        pizza += oven_temperature  # This is a very crude representation of baking
        baking_time += 1

    return pizza
    
# Good Code

def make_pizza():
    prepare_dough()
    apply_tomato_sauce()
    add_cheese()
    bake_in_oven()

def prepare_dough():
    mix_ingredients()
    knead_dough()
    roll_out_dough()

def mix_ingredients():
    # Details of mixing flour, water, yeast etc.

def knead_dough():
    # Details of kneading the dough to make it smooth

def roll_out_dough():
    # Details of rolling the dough to the right thickness

def apply_tomato_sauce():
    # Details of applying tomato sauce on the rolled out dough

def add_cheese():
    # Details of adding cheese on the tomato sauce

def bake_in_oven():
    preheat_oven()
    set_timer()
    place_pizza_in_oven()

def preheat_oven():
    # Details of preheating the oven to the right temperature

def set_timer():
    # Details of setting the timer to the correct baking time

def place_pizza_in_oven():
    # Details of placing the pizza in the oven safely
```

```python
# Good
for user in users_with_pending_tasks:
    if user.active_count < user.quota_limit:
        task = user.queue.pop()
        assign_to_available_worker(task)
        user.active_count += 1

# Better
for user in users_with_pending_tasks:
    if user.active_count < user.quota_limit:
        schedule_user(user)

def schedule_user(user):
    task = user.queue.pop()
    assign_to_available_worker(task)
    user.active_count += 1

```

---

## Library / Framework First Assumption

Unless explicitly stated otherwise, **always assume that we are building a library or framework, not an application or script**.

This implies the following non-negotiable rules:

* The user **cannot modify library source code or directories**
* The user **cannot add files, types, or logic inside the library**
* All customization must happen **outside the library**, via:

  * dependency injection
  * composition
  * traits / interfaces
  * callbacks or configuration objects

A design is **invalid** if:

* extending behavior requires editing library files
* new user-defined concepts require adding enum values
* functionality is centralized in “registry” files that must be modified

The library must be:

* open for extension
* closed for modification
* usable in unknown future contexts

Configuration rules:

* everything that can be configured **must be configurable**
* defaults are allowed, hard-coded decisions are not
* CLI arguments are adapters, **not core configuration**

Script vs library separation:

* core logic must live in reusable library functions/classes
* scripts must only parse arguments, call the library, and serialize results

When in doubt, **prefer designs that maximize user freedom and long-term extensibility**, even at the cost of slightly higher initial complexity.


# General
We work in .venv python environment. not global one. Fedora linux + CPU: 11th Gen Intel i9-11900H (16) + NVIDIA GeForce RTX 3060 Mobile 6 Gb VRAM + 64 GB RAM
Do not write useless comments and prints, no vibe-coded bullshit as emojis and checkmarks in code, use what everybody can easely type with keyboard. Code should be easely to understand and extend. Keep everything simple
When thinking about architecture: you are a staff quantitative developer, PhD in Computer Science and 10 years in Software Engineering.
Overall you are a senior Quantitative Researcher and Developer at High- and Medium- Frequency trading

Do not forget to use dev tools proactively
Read anti-ai-slop first ALWAYS and write only things satisfying it.

You are senior quant researcher and kaggle competition grandmaster, years of experience in Jane Street