# P3M Document Relationships Guide

[document-relationships.mermaid](./P3M%20Document%20Relationships%20Guide-assets/document-relationships.mermaid)

![ScreenShot 2024-11-01 at 15.01.23@2x.png](./P3M%20Document%20Relationships%20Guide-assets/ScreenShot%202024-11-01%20at%2015.01.23@2x.png)

[document-cycle.mermaid](./P3M%20Document%20Relationships%20Guide-assets/document-cycle.mermaid)

![ScreenShot 2024-11-01 at 15.02.03@2x.png](./P3M%20Document%20Relationships%20Guide-assets/ScreenShot%202024-11-01%20at%2015.02.03@2x.png)

## 1\. Hierarchical Relationships

### 1\.1 Strategic Flow

```
Portfolio Level
├── Strategic Plan
│   ├── Program Charter
│   │   └── Project Charter
│   └── Program Business Case
│       └── Project Business Case
└── Investment Strategy
    ├── Program Blueprint
    │   └── Project Requirements
    └── Portfolio Prioritization
        └── Project Selection
```

### 1\.2 Governance Flow

```
Portfolio Level
├── Portfolio Management Framework
│   ├── Program Governance Framework
│   │   └── Project Governance
│   └── Change Control Framework
│       └── Project Change Management
└── Risk Framework
    ├── Program Risk Management
    └── Project Risk Management
```

### 1\.3 Benefits Flow

```
Portfolio Level
├── Benefits Realization Framework
│   ├── Program Benefits Plan
│   │   └── Project Success Criteria
│   └── Benefits Tracking
│       └── Project Metrics
```

## 2\. Document Dependencies

### 2\.1 Definition Documents Flow

```
Portfolio → Program → Project
├── Vision/Strategy → Objectives → Deliverables
├── Investment → Business Case → Requirements
└── Scope → Blueprint → WBS
```

### 2\.2 Control Documents Flow

```
Portfolio → Program → Project
├── Governance → Management → Execution
├── Risk → Issues → Actions
└── Benefits → Outcomes → Outputs
```

## 3\. Key Relationships

### 3\.1 Strategic Alignment

- Portfolio Strategic Plan defines overall direction

- Program Charter aligns with strategic objectives

- Project Charter delivers specific strategic elements

### 3\.2 Investment Management

- Portfolio Investment Strategy sets criteria

- Program Business Case justifies investment

- Project Plans detail resource utilization

### 3\.3 Scope Management

- Portfolio Charter defines boundaries

- Program Blueprint details solution scope

- Project WBS breaks down deliverables

### 3\.4 Change Control

- Portfolio Framework sets change thresholds

- Program Change Control defines process

- Project Change Management implements controls

## 4\. Document Updates and Feedback

### 4\.1 Bottom-up Reporting

```
Project → Program → Portfolio
├── Status → Health → Performance
├── Issues → Risks → Strategic Risks
└── Metrics → Benefits → Value
```

### 4\.2 Top-down Direction

```
Portfolio → Program → Project
├── Strategy → Objectives → Goals
├── Standards → Controls → Procedures
└── Priorities → Focus → Deliverables
```

## 5\. Temporal Relationships

### 5\.1 Planning Phase

```
Portfolio Level
├── Strategic Plan
└── Investment Strategy
    ↓
Program Level
├── Business Case
└── Blueprint
    ↓
Project Level
├── Charter
└── Requirements
```

### 5\.2 Execution Phase

```
Portfolio Level
├── Performance Monitoring
└── Risk Management
    ↓
Program Level
├── Benefits Tracking
└── Change Control
    ↓
Project Level
├── Delivery Management
└── Issue Resolution
```

### 5\.3 Closure Phase

```
Project Level
├── Deliverable Acceptance
└── Lessons Learned
    ↑
Program Level
├── Benefits Realization
└── Capability Delivery
    ↑
Portfolio Level
├── Value Assessment
└── Strategic Alignment
```

## 6\. Key Integration Points

### 6\.1 Decision Gates

```
Portfolio Review
├── Program Stage Gates
│   └── Project Phase Gates
└── Investment Reviews
    └── Business Case Updates
```

### 6\.2 Risk and Issue Management

```
Portfolio Risks
├── Program Risks
│   └── Project Risks
└── Risk Aggregation
    └── Mitigation Strategies
```

### 6\.3 Resource Management

```
Portfolio Capacity
├── Program Resources
│   └── Project Teams
└── Resource Optimization
    └── Skills Management
```