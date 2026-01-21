# Formal Ontology Representations

## Frameworks

### First-order Logic

### OWL

### Description Logic

<https://en.wikipedia.org/wiki/Description_logic>

Models with T-Box and A-Box statements

Used by the [Virtual Ontology ](https://github.com/mcfitzgerald/virtual-ontology)guy (

[Medium - Whither Ontologies? Palantir-lite with Claude Code by Michael Fitzgerald](https://medium.com/@michael.craig.fitzgerald/whither-ontologies-d871bd3a8098)

Example:

```bash
# Ontology Specification (T-Box/R-Box formalization) - truncated
ontology:
  name: "Manufacturing Execution System"
  domain: "Production Operations"
  
classes:
  Equipment:
    description: "Physical assets on production line"
    attributes: 
      - efficiency
      - upstream_dependencies
      - maintenance_schedule
    
  DowntimeEvent:
    description: "Production stoppage with reason code"
    attributes:
      - reason_code
      - duration
      - cascade_impact

relationships:
  is_upstream_of:
    domain: Equipment
    range: Equipment
    properties: 
      - cascade_delay: "typical seconds before downstream impact"
      - impact_correlation: "probability of cascade"
      
business_rules:
  material_starvation:
    when: "upstream equipment fails"
    then: "downstream shows UNP-MAT code"
    delay: "30-300 seconds typically"
```

## Comparison

| FOL | OWL | DL | Examples | 
|---|---|---|---|
| constant | individual | individual | Mickey Mouse, Walter Elias Mouse, Paris, France, etc. | 
| unary predicate | class | concept | (Being a) person, a city, a country, etc. | 
| binary predicate | property | role | father of, located in, etc. | 

