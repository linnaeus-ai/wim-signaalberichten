from pydantic import BaseModel, Field
from typing import List, Optional


class RetrieveSchemaOrgStructuredOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Een korte verklaring (één zin) waarom deze schema.org class is gekozen in de vorm: 'In de gegeven tekst, de beste schema.org class voor class [input class] is [selected class] omdat [reden].'",
    )
    selected_class: str = Field(
        ...,
        description="De naam van de geselecteerde schema.org class.",
    )
    selected_number: int = Field(
        ...,
        description="The number (1-5) of the selected schema.org class from the candidates list.",
        ge=1,
        le=5,
    )


class AddLabelsStructuredOutput(BaseModel):
    onderwerp_labels: List[str] = Field(
        ...,
        description="Lijst van onderwerp labels (waar gaat het over) die van toepassing zijn op de tekst. Kies 'No subtopic found' als er geen passend label is.",
        min_items=1,
    )
    beleving_labels: List[str] = Field(
        ...,
        description="Lijst van beleving labels (hoe wordt de service ervaren) die van toepassing zijn op de tekst. Kies 'No subtopic found' als er geen passend label is.",
        min_items=1,
    )


class Prefix(BaseModel):
    prefix: str
    uri: str


class ClassDef(BaseModel):
    name: str
    subclass_of: Optional[str] = None


class ObjectProperty(BaseModel):
    name: str
    domain: str
    range: str


class DatatypeProperty(BaseModel):
    name: str
    domain: str
    range: str  # usually "xsd:string", etc.


class PropertyItem(BaseModel):  # Renamed from Attributes
    name: str
    datatype: str
    min_value: Optional[float] = None  # For numeric types or sizes
    max_value: Optional[float] = None  # For numeric types or sizes
    example: Optional[str] = None  # Example value for the attribute
    is_required: bool = False  # Indicates if the attribute is mandatory
    is_unique: bool = False  # Indicates if the attribute must be unique

    def to_description_string(self) -> str:
        parts = [f"`{self.name}`: {self.datatype}"]
        if self.example is not None:
            # Ensure example string is quoted if it contains spaces or special chars,
            # and escape internal quotes if necessary.
            # Simple quoting for now:
            example_str = self.example
            if '"' in example_str and "'" not in example_str:
                example_str = f"'{example_str}'"
            else:
                example_str = f'"{example_str}"'
            parts.append(f"Example: {example_str}")

        if self.min_value is not None:
            min_val_str = (
                f"{self.min_value:g}"  # Use :g for general format (int if whole)
            )
            parts.append(f"Min: {min_val_str}")
        if self.max_value is not None:
            max_val_str = (
                f"{self.max_value:g}"  # Use :g for general format (int if whole)
            )
            parts.append(f"Max: {max_val_str}")

        # The requested string format doesn't include is_required or is_unique
        # If needed, they can be added here.
        # if self.is_required:
        #     parts.append("(required)")
        # if self.is_unique:
        #     parts.append("(unique)")

        return " ".join(parts)


# RelationshipProperties class is removed as PropertyItem can be used.


class NodeSchema(BaseModel):
    node_type: str  # e.g., "Movie", "Person"
    properties: List[PropertyItem]


class EdgeSchema(BaseModel):
    edge_type: str  # e.g., "ACTED_IN", "REVIEWED"
    properties: List[PropertyItem]


class PropertyGraphSchema(BaseModel):
    nodes: List[NodeSchema]  # Replaces node_properties: List[Attributes]
    edges: List[
        EdgeSchema
    ]  # Replaces relationship_properties: List[RelationshipProperties]
    relationships: List[str]  # Cypher-like syntax for relationships

    def __str__(self) -> str:
        lines = []

        if self.nodes:
            lines.append("Node properties:")
            for node_schema in self.nodes:
                lines.append(f"- **{node_schema.node_type}**")
                for prop_item in node_schema.properties:
                    lines.append(f"  - {prop_item.to_description_string()}")
            if self.edges or self.relationships:  # Add spacing if other sections follow
                lines.append("")

        if self.edges:
            lines.append("Relationship properties:")
            for edge_schema in self.edges:
                lines.append(f"- **{edge_schema.edge_type}**")
                for prop_item in edge_schema.properties:
                    lines.append(f"  - {prop_item.to_description_string()}")
            if self.relationships:  # Add spacing if other sections follow
                lines.append("")

        if self.relationships:
            lines.append("The relationships:")
            for rel_link in self.relationships:
                lines.append(rel_link)

        return "\n".join(lines)


class OwlSchema(BaseModel):
    prefixes: List[Prefix]
    classes: List[ClassDef]
    object_properties: List[ObjectProperty]
    datatype_properties: List[DatatypeProperty]

    def to_turtle(self) -> str:
        lines = []

        for prefix in self.prefixes:
            lines.append(f"@prefix {prefix.prefix}: <{prefix.uri}> .")

        lines.append("")  # spacing

        for cls in self.classes:
            line = f":{cls.name} rdf:type owl:Class"
            if cls.subclass_of:
                line += f" ;\n         rdfs:subClassOf :{cls.subclass_of}"
            line += " ."
            lines.append(line)

        lines.append("")  # spacing

        for prop in self.object_properties:
            lines.append(
                f":{prop.name} rdf:type owl:ObjectProperty ;\n             rdfs:domain :{prop.domain} ;\n             rdfs:range :{prop.range} ."
            )

        lines.append("")  # spacing

        for prop in self.datatype_properties:
            lines.append(
                f":{prop.name} rdf:type owl:DatatypeProperty ;\n             rdfs:domain :{prop.domain} ;\n             rdfs:range {prop.range} ."
            )

        return "\n".join(lines)


class RdfTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    is_literal: bool = False
    literal_type: Optional[str] = None  # e.g., "xsd:string"


class RdfIndividual(BaseModel):
    uri: str
    types: List[str]  # e.g., ["schema:Persoon", "schema:President"]
    attributes: List[RdfTriple] = []  # Datatype properties
    relations: List[RdfTriple] = []  # Object properties


class KnowledgeGraph(BaseModel):
    prefixes: List[Prefix]
    classes: List[str]  # Class URIs like "schema:Persoon"
    individuals: List[RdfIndividual]
    standalone_triples: List[RdfTriple] = []  # For triples not grouped with individuals

    def to_turtle(self) -> str:
        """Generate Turtle syntax from the structured representation"""
        lines = []

        # Prefixes
        for prefix in self.prefixes:
            lines.append(f"@prefix {prefix.prefix}: <{prefix.uri}> .")

        lines.append("")  # spacing

        # Individuals with their types, attributes and relations
        for individual in self.individuals:
            # Open individual block
            lines.append(f"{individual.uri}")

            # Types - ensure each class has a prefix if it doesn't already have one
            if individual.types:
                # Add default prefix ":" to classes that don't have an explicit prefix
                prefixed_types = []
                for t in individual.types:
                    if ":" not in t:
                        prefixed_types.append(f":{t}")
                    else:
                        prefixed_types.append(t)
                type_str = ", ".join(prefixed_types)
                lines.append(f"    rdf:type {type_str} ;")

            # Attributes (datatype properties)
            for attr in individual.attributes:
                obj_value = attr.object
                if attr.is_literal:
                    if attr.literal_type:
                        obj_value = f'"{obj_value}"^^{attr.literal_type}'
                    else:
                        obj_value = f'"{obj_value}"'

                # Add prefix to predicate if missing
                predicate = attr.predicate
                if ":" not in predicate:
                    predicate = f":{predicate}"

                lines.append(f"    {predicate} {obj_value} ;")

            # Relations (object properties)
            for rel in individual.relations:
                # Add prefix to predicate if missing
                predicate = rel.predicate
                if ":" not in predicate:
                    predicate = f":{predicate}"

                lines.append(f"    {predicate} {rel.object} ;")

            # Close individual block (replace last semicolon with period)
            if len(lines) > 0 and lines[-1].endswith(" ;"):
                lines[-1] = lines[-1][:-2] + " ."
            else:
                lines.append("    .")

            lines.append("")  # spacing

        # Standalone triples (not grouped with individuals)
        for triple in self.standalone_triples:
            obj_value = triple.object
            if triple.is_literal:
                if triple.literal_type:
                    obj_value = f'"{obj_value}"^^{triple.literal_type}'
                else:
                    obj_value = f'"{obj_value}"'

            # Add prefix to predicate if missing
            predicate = triple.predicate
            if ":" not in predicate:
                predicate = f":{predicate}"

            lines.append(f"{triple.subject} {predicate} {obj_value} .")

        return "\n".join(lines)
