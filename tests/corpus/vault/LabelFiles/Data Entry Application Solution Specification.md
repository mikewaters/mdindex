# Data Entry Application Solution Specification

## System Architecture

### Overview

The application will follow a split architecture pattern with a React frontend and SQLite backend. The system will be built as a single-page application (SPA) with real-time markdown editing capabilities and contextual data entry forms.

### Component Structure

```
src/
├── components/
│   ├── MarkdownEditor/
│   │   ├── Editor.tsx
│   │   ├── LineSelector.tsx
│   │   └── ProcessingIndicator.tsx
│   ├── DataEntryForm/
│   │   ├── PredicateSelector.tsx
│   │   ├── ObjectTypeSelector.tsx
│   │   ├── ObjectSelector.tsx
│   │   └── StatementBuilder.tsx
│   └── Layout/
│       ├── SplitPane.tsx
│       └── FocusManager.tsx
├── hooks/
│   ├── useMarkdownProcessor.ts
│   ├── useStatementBuilder.ts
│   └── useDatabaseOperations.ts
├── services/
│   ├── database.ts
│   └── statementProcessor.ts
└── types/
    ├── Statement.ts
    ├── Predicate.ts
    └── Object.ts
```

## Database Schema

### Tables

#### predicates

```sql
CREATE TABLE predicates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### object_types

```sql
CREATE TABLE object_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### predicate_object_type_mappings

```sql
CREATE TABLE predicate_object_type_mappings (
    predicate_id INTEGER,
    object_type_id INTEGER,
    FOREIGN KEY (predicate_id) REFERENCES predicates(id),
    FOREIGN KEY (object_type_id) REFERENCES object_types(id),
    PRIMARY KEY (predicate_id, object_type_id)
);
```

#### objects

```sql
CREATE TABLE objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_id INTEGER,
    value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (type_id) REFERENCES object_types(id)
);
```

#### statements

```sql
CREATE TABLE statements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate_id INTEGER,
    object_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (predicate_id) REFERENCES predicates(id),
    FOREIGN KEY (object_id) REFERENCES objects(id)
);
```

## Key Components

### MarkdownEditor

The MarkdownEditor component will be built on top of CodeMirror, with the following customizations:

1. Line Processing

   - Custom line decorations for processed vs unprocessed lines

   - Line-level click handlers for selection

   - Custom gutters for processing status indicators

2. Keyboard Navigation

   - Custom key bindings for line navigation

   - Command/Ctrl + Enter handler for activating right panel

   - Arrow key handlers for line navigation

### DataEntryForm

The form will be implemented as a controlled component with the following features:

1. Dynamic Field Loading

   - Predicate selection affects available object types

   - Object type selection affects searchable objects

2. Focus Management

   - Tab trap within form until submission

   - Explicit escape handling for returning to editor

   - Form field focus order optimization

## State Management

### Local Component State

```typescript
interface EditorState {
    content: string;
    selectedLine: number | null;
    processedLines: Set<number>;
}

interface FormState {
    subject: string;
    predicateId: number | null;
    objectTypeId: number | null;
    objectId: number | null;
}
```

### Database Operations

```typescript
interface DatabaseService {
    getPredicates(): Promise<Predicate[]>;
    getObjectTypes(): Promise<ObjectType[]>;
    getObjectsByType(typeId: number): Promise<Object[]>;
    createObject(type: number, value: string): Promise<number>;
    createStatement(subject: string, predicateId: number, objectId: number): Promise<number>;
}
```

## User Interface Specifications

### Keyboard Shortcuts

- Cmd/Ctrl + Enter: Activate right panel for current line

- Tab: Navigate through form fields

- Escape: Return focus to editor

- Arrow Up/Down: Navigate between lines in editor

- Cmd/Ctrl + S: Submit current statement

### Visual Indicators

1. Line Processing Status

   - Unprocessed: No indicator

   - Processed: Green checkmark in gutter

   - Selected: Highlighted background

   - Error: Red indicator in gutter

2. Form Status

   - Loading: Spinner overlay on form

   - Validation Errors: Red outline on fields

   - Success: Green flash on submission

## Error Handling

### Database Operations

```typescript
class DatabaseError extends Error {
    constructor(
        message: string,
        public operation: string,
        public originalError: Error
    ) {
        super(message);
    }
}
```

### Form Validation

```typescript
interface ValidationError {
    field: string;
    message: string;
}

type ValidationResult = ValidationError[] | null;
```

## Performance Considerations

1. Editor Optimization

   - Debounced line processing

   - Virtualized line rendering for large documents

   - Memoized line decorations

2. Database Operations

   - Prepared statements for frequent queries

   - Connection pooling

   - Batched updates for multiple statements

3. Form Rendering

   - Lazy loading of object lists

   - Debounced search operations

   - Cached predicate/object type relationships

## Testing Strategy

1. Unit Tests

   - Component rendering

   - Form validation

   - Database operations

   - State management

2. Integration Tests

   - Editor-form interaction

   - Database integration

   - Keyboard navigation

3. End-to-End Tests

   - Complete data entry workflow

   - Error handling scenarios

   - Performance benchmarks

## Development Workflow

1. Phase 1: Core Infrastructure

   - Database schema setup

   - Basic component structure

   - State management implementation

2. Phase 2: Editor Implementation

   - CodeMirror integration

   - Line processing

   - Keyboard navigation

3. Phase 3: Form Implementation

   - Dynamic field relationships

   - Object creation/selection

   - Validation

4. Phase 4: Integration

   - State synchronization

   - Error handling

   - Performance optimization

5. Phase 5: Testing and Documentation

   - Test suite implementation

   - Performance testing

   - Documentation generation