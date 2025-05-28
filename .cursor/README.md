# Cursor Configuration Directory

This directory contains configuration files for the Cursor IDE to enhance the development experience with Spark Stacker.

## Files

### repository_rules.json

This file provides a comprehensive overview of the repository structure, architecture, and development workflow. It helps Cursor AI agents understand:

- Repository structure and package organization
- Component architecture and data flow
- Development prerequisites and setup instructions
- Extension points for customizing the platform
- Configuration file locations

The information in this file enables Cursor to provide more accurate and contextual assistance when working with this codebase.

### mcp.json

Contains Cursor MCP (Multi-Component Protocol) server configuration, which supports NX integration and improves the IDE's understanding of the monorepo structure.

## Purpose

These configuration files help Cursor AI agents and extensions:

1. Quickly understand the repository structure
2. Provide more accurate code suggestions
3. Navigate efficiently between related components
4. Understand data flow and architectural decisions
5. Offer contextually appropriate guidance based on which part of the codebase you're working with

## Updating

If significant changes are made to the repository structure or architecture, consider updating the `rules.json` file to ensure Cursor continues to provide accurate assistance.
