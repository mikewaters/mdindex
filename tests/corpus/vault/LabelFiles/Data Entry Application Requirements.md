# Data Entry Application Requirements

## Overview

A web-based application for rapid data entry that allows users to paste or type blocks of structured text and associate it with metadata and relationships. The application uses a split-screen interface with a Markdown editor on the left and a contextual data entry form on the right.

## Core Features

### Left Panel - Markdown Editor

1. Text Entry

   - Accept pasted or typed Markdown text

   - Support standard Markdown syntax highlighting

   - Provide line-level interaction for data lines

   - Visual indicators for processed vs unprocessed data lines

   - Clear distinction between structural Markdown (headers, lists) and data lines

2. Text Selection

   1. Entire Line Selection

      1. Click or move the editor cursor to select individual lines to be processed

      2. Visual indication of currently selected line

      3. Ability to distinguish between data-containing lines and structural Markdown

      4. Support for keyboard navigation between lines

   2. Inline Selection

      1. Select a block of text with the cursor or mouse to be processed

### Right Panel - Data Entry Form

The selection made in the Left Panel (the line or selection) will serve as the “Subject” in a “Subject, Predicate, Object” semantic triple. This is called a “Statement”.  The user can select the remaining Predicate and Object and cause that statement to be persisted to the database.

1. Predicate Selection

   1. There is a predefined list of supported predicates, stored in a database table.

2. Object type selection

   1. There is a predefined list of object types, and the allowed predicates that can apply to the object.

   2. The user can select an object type, which can serve as a filter for searching for an object, or to inform the type of a new object being created

3. Object Selection

   1. The user can search for existing objects of the given object type selected, and when found the user can persist the Statement to the database

   2. If an existing object is not found, the user can create a new object with the object type selected. This new object will be persisted to the database along with the Statement.

## Data Types

### Statements

The user assembles some string of text to either a new object, or a relationship to an existing object, using Statements. The text string is the Subject of the statement.

#### Statements resolving to a new Object

Examples:

- Subject + "is a" + Object Type (predicate expressing type/classification)

#### Statements resolving to a new Relationship

Examples:

- Subject + "is related to" + Object Instance (predicate expressing general relationship)

- Subject + "is a child of" + Object Instance (predicate expressing hierarchical relationship)

- Subject + "is a parent of" + Object Instance (predicate expressing hierarchical relationship)

### Predicates

Available predicates (more to be added in the future):

- Is a

- Is related to

- Is a child of

- Is the parent of

### Object Types

Available types (more to be added in the future):

- Effort

- Outcome

Example:

- Subject “is a” Outcome

### Objects

Given an Object Type, Object instances will be stored in a database table for that object type.

Examples:

- Efforts table

- Outcomes table

## User Experience Requirements

### Navigation and Input Flow

1. Keyboard-First Navigation

   - Left pane is default focus on load

   - Full text editing and navigation capabilities in Markdown editor

   - Explicit keyboard action (e.g., Cmd+Enter) to activate right pane for current line or current selection

   - Same keyboard action used for both:

      - Initializing metadata entry for new lines

      - Loading existing metadata for previously processed lines

   - No automatic loading of metadata when navigating to processed lines

   - Tab navigation through form fields in right pane

   - Keyboard shortcut to return focus to left pane

   - Arrow keys for line navigation when in left pane

2. Visual Feedback

   - Clear indication of selected line

   - Status indicators for processed lines - those which have already been processed as statements

   - Error states for invalid data

   - Loading states for async operations

### Data Entry Flow

1. Line Processing

   - Navigate to line in left panel using keyboard

   - Use dedicated keyboard shortcut to activate right panel

   - Right panel behavior:

      - For unprocessed lines: Initialize empty form

      - For processed lines: Load existing metadata only on explicit activation

   - Tab through form fields for rapid data entry

   - Submit form with keyboard shortcut

2. Focus Management

   - Clear visual indication of which pane has focus

   - Predictable focus movement between panes

   - Focus trap in right pane form until submission or explicit exit

   - Preserve cursor position in left pane when returning focus

### Data Persistence

1. Save Behavior

   - Final submission for completed entries

## Technical Requirements

### Frontend

1. Framework

   - React-based implementation in Typescript

   - CodeMirror for Markdown editing

   - Responsive design for different screen sizes

2. State Management

   - Local component state for current entry

   - Minimal global state

   - Efficient handling of large text documents

### Database

1. Local SQLIte database

   
