# Cursor Configuration & Rules Directory

This directory contains comprehensive configuration and coding rules for the Spark Stacker project, designed to enhance the development experience with Cursor IDE through intelligent AI assistant integration.

## 🎯 **Directory Overview**

### 📁 **`rules/`** - Coding Rules & Guidelines

Organized collection of development standards, architectural principles, and best practices with intelligent trigger mechanisms for optimal AI assistance.

### ⚙️ **`mcp.json`** - MCP Server Configuration

Cursor MCP (Multi-Component Protocol) server configuration supporting NX integration and monorepo structure understanding.

### 📄 **`README.md`** - This comprehensive guide

Complete documentation for all Cursor configurations and coding rules.

---

## 📊 **Complete Rules Reference**

| Rule                                                   | Purpose                                                | Configuration       | Trigger                                                                                                                    |
| ------------------------------------------------------ | ------------------------------------------------------ | ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **🏗️ Architecture**                                    |                                                        |                     |                                                                                                                            |
| `rules/architecture/project-structure.mdc`             | Project organization and monorepo structure guidelines | **Agent Requested** | _"Project structure guidelines and organization principles for the Spark Stacker monorepo"_                                |
| `rules/architecture/symbol-conversion.mdc`             | Exchange symbol conversion patterns and standards      | **Agent Requested** | _"Guide for symbol conversion architecture between standard format and exchange-specific formats in trading applications"_ |
| `rules/architecture/strategy-indicator-separation.mdc` | Strategy-indicator relationship architecture           | **Agent Requested** | _"Architecture principles for separating strategy logic from indicator implementations in trading systems"_                |
| **⚙️ Configuration**                                   |                                                        |                     |                                                                                                                            |
| `rules/configuration/position-sizing.mdc`              | Position sizing configuration patterns and validation  | **Auto Attached**   | `config.json, **/config/**, **/*config*.py, **/*config*.json`                                                              |
| **🛠️ Development**                                     |                                                        |                     |                                                                                                                            |
| `rules/development/file-naming.mdc`                    | File and folder naming conventions across the project  | **Auto Attached**   | `*.py, *.ts, *.md, **/tests/**, **/docs/**`                                                                                |
| `rules/development/python-best-practices.mdc`          | Python coding standards, testing, and best practices   | **Auto Attached**   | `*.py, requirements.txt, pytest.ini, **/.venv/**`                                                                          |
| **🧪 Testing**                                         |                                                        |                     |                                                                                                                            |
| `rules/testing/spark-stacker-testing-guide.mdc`        | Comprehensive testing framework and guidelines         | **Auto Attached**   | `**/tests/**, test_*.py, *_test.py, pytest.ini, conftest.py`                                                               |
| **🔧 Tools**                                           |                                                        |                     |                                                                                                                            |
| `rules/tools/nx-rules.mdc`                             | NX workspace management and monorepo guidelines        | **Auto Attached**   | `nx.json, project.json, package.json, workspace.json`                                                                      |
| `rules/tools/use-yarn.mdc`                             | Yarn package manager usage and conventions             | **Auto Attached**   | `package.json, yarn.lock, .yarnrc*`                                                                                        |
| **🚧 Troubleshooting**                                 |                                                        |                     |                                                                                                                            |
| `rules/troubleshooting/common-errors.mdc`              | Common error patterns, debugging, and solutions        | **Agent Requested** | _"Common errors and debugging solutions for Spark Stacker development and configuration issues"_                           |
| **🔄 Workflow**                                        |                                                        |                     |                                                                                                                            |
| `rules/workflow/checklist-workflow.mdc`                | Checklist-driven development workflow methodology      | **Agent Requested** | _"Checklist-driven development workflow for focused, incremental progress tracking"_                                       |
| `rules/workflow/commit-conventions.mdc`                | Git commit message standards and conventions           | **Agent Requested** | _"Git commit message conventions and standards for consistent version control history"_                                    |

## 🎛️ **Configuration Types Explained**

### **Always**

Rules that are **always active** for every AI conversation.

- Applied to fundamental, universal guidelines
- Currently no rules use this setting

### **Auto Attached**

Rules that **automatically activate** when working with specific file types.

- Triggered by file patterns (e.g., `*.py`, `package.json`)
- Perfect for development standards and tool-specific guidelines
- **6 rules** use this configuration

### **Agent Requested**

Rules that are **available on demand** when the AI determines they're relevant.

- Activated by descriptive triggers about the rule's purpose
- Ideal for specialized knowledge, architecture guidance, and troubleshooting
- **6 rules** use this configuration

## 🎯 **Usage Guidelines**

### **For Development Work**

These rules automatically activate when editing relevant files:

- **Python development** → `python-best-practices.mdc` + `file-naming.mdc`
- **Testing** → `spark-stacker-testing-guide.mdc`
- **Configuration changes** → `position-sizing.mdc`
- **Package management** → `use-yarn.mdc`
- **NX workspace** → `nx-rules.mdc`

### **For Architecture & Design**

Request these rules when planning or discussing:

- Project structure decisions → Ask about _"project organization"_
- Symbol handling → Ask about _"symbol conversion architecture"_
- Strategy design → Ask about _"strategy-indicator separation"_

### **For Problem Solving**

Request these rules when encountering issues:

- Development workflow → Ask about _"checklist workflow"_
- Git standards → Ask about _"commit conventions"_
- Error debugging → Ask about _"common errors and solutions"_

## 📁 **Rules Folder Organization**

### 🏗️ **Architecture** (3 rules)

High-level design principles and architectural patterns

- **Configuration**: Agent Requested
- **When to use**: System design, refactoring, architectural decisions

### ⚙️ **Configuration** (1 rule)

Configuration file patterns and validation

- **Configuration**: Auto Attached
- **When to use**: Automatically when editing config files

### 🛠️ **Development** (2 rules)

Core development practices and standards

- **Configuration**: Auto Attached
- **When to use**: Automatically during active development

### 🧪 **Testing** (1 rule)

Testing frameworks, patterns, and best practices

- **Configuration**: Auto Attached
- **When to use**: Automatically when working with test files

### 🔧 **Tools** (2 rules)

Tool-specific usage and configuration guidelines

- **Configuration**: Auto Attached
- **When to use**: Automatically when working with tool config files

### 🚧 **Troubleshooting** (1 rule)

Error resolution and debugging guidance

- **Configuration**: Agent Requested
- **When to use**: When encountering errors or debugging issues

### 🔄 **Workflow** (2 rules)

Development process and methodologies

- **Configuration**: Agent Requested
- **When to use**: For process guidance and workflow questions

## 🔗 **Cross-References**

Rules frequently reference:

- **Project files**: `mdc:packages/spark-app/...`
- **Configuration**: `mdc:packages/shared/config.json`
- **Documentation**: `mdc:packages/shared/docs/...`
- **Other rules**: `mdc:.cursor/rules/folder/rule.mdc`

## 📝 **Rule Format Standards**

All rules follow consistent markdown format:

- **Clear headings** with emoji indicators
- **Code examples** with proper syntax highlighting
- **Cross-references** using `mdc:` syntax
- **Practical examples** and real-world use cases
- **Error patterns** and solutions where applicable

## 🚀 **Adding New Rules**

### Step 1: Determine Configuration Type

- **Auto Attached**: Development standards, tool guidelines, frequent patterns
- **Agent Requested**: Specialized knowledge, architecture, troubleshooting

### Step 2: Choose Folder

- Review table above to find appropriate category
- Create new folder if no existing category fits

### Step 3: Create Rule File

- Use kebab-case naming (e.g., `new-feature-guide.mdc`)
- Follow existing format standards
- Include comprehensive examples

### Step 4: Update Documentation

- Add entry to the rules table in this README
- Update folder counts and descriptions
- Test configuration works as expected

## 🔧 **Technical Configuration**

### MCP Server Integration

The `mcp.json` file configures:

- **NX workspace integration** for monorepo understanding
- **Component relationship mapping** for intelligent navigation
- **Build system awareness** for development workflow optimization

### AI Assistant Benefits

This configuration enables Cursor AI to:

1. **Understand repository structure** automatically
2. **Provide contextual code suggestions** based on active files
3. **Navigate efficiently** between related components
4. **Offer architectural guidance** when appropriate
5. **Apply coding standards** consistently across the project

## 🔄 **Rule Maintenance**

### Regular Review Process

- **Monthly**: Review rule relevance and accuracy
- **Per Release**: Update examples and references
- **Per Architecture Change**: Update architectural rules

### Quality Standards

- **Single source of truth** - No duplicate information
- **Living documentation** - Keep examples current
- **Clear ownership** - Maintain accountability
- **Version control** - Track changes with descriptive commits

### Updating Configuration

If significant changes are made to the repository structure or architecture:

1. **Update relevant rule files** with new patterns and examples
2. **Modify MCP configuration** if component relationships change
3. **Test AI assistant behavior** to ensure optimal assistance
4. **Update this documentation** to reflect changes

## 📈 **Statistics & Coverage**

- **Total Rules**: 12 files across 7 categories
- **Auto Attached**: 6 rules (50%) - Active during development
- **Agent Requested**: 6 rules (50%) - Available on demand
- **Always Active**: 0 rules (0%) - None currently
- **Coverage Areas**:
  - ✅ Development Standards
  - ✅ Architecture Principles
  - ✅ Testing Guidelines
  - ✅ Tool Configurations
  - ✅ Workflow Processes
  - ✅ Configuration Patterns
  - ✅ Troubleshooting Support

## 🎁 **Benefits**

### ✅ **Developer Experience**

- **Context-aware assistance** based on current work
- **Consistent coding standards** applied automatically
- **Quick access** to relevant architectural guidance
- **Reduced cognitive load** through intelligent rule activation

### ✅ **Code Quality**

- **Enforced naming conventions** across the project
- **Standardized patterns** for indicators, connectors, strategies
- **Comprehensive testing guidance** for all components
- **Architecture compliance** through guided development

### ✅ **Team Coordination**

- **Shared understanding** of project structure and patterns
- **Consistent workflows** across all team members
- **Clear troubleshooting** resources for common issues
- **Documented processes** for all development activities

### ✅ **Maintainability**

- **Scalable organization** that grows with the project
- **Version-controlled standards** with change tracking
- **Cross-referenced documentation** for easy navigation
- **Future-proof architecture** with extensible patterns

---

**Maintained by**: Development Team
**Last Updated**: 2024-12-28
**Configuration Version**: v2.0 (Comprehensive table-based organization)
**Total Configuration Files**: 3 (README.md, mcp.json, rules/)
