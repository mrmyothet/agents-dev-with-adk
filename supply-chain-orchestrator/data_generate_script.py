import csv
import random

# Configuration
NUM_ASSEMBLIES = 50
COMPONENTS_PER_BIKE = 8
OUTPUT_ASSEMBLY = "assemblies.csv"
OUTPUT_BOM = "bom_data.csv"

categories = ["Road", "Mountain", "City", "Electric", "Gravel"]
comp_types = [
    "Frame",
    "Drivetrain",
    "Wheelset",
    "Brake-Set",
    "Saddle",
    "Handlebar",
    "Tires",
    "Fork",
]

# 1. Generate Assemblies
assemblies = []
for i in range(1, NUM_ASSEMBLIES + 1):
    sku = f"BK-{random.choice(categories)[:3].upper()}-{i:02d}"
    # Scenario Logic: 30% In Stock, 70% Out of Stock (to force agent to check BOM)
    stock = random.randint(5, 20) if random.random() > 0.7 else 0
    assemblies.append(
        {
            "SKU": sku,
            "Name": f"{random.choice(['Aero', 'Trail', 'Apex', 'Peak'])} {i}",
            "Stock": stock,
        }
    )

# 2. Generate BOM & Components
bom_rows = []
for bike in assemblies:
    for c_type in comp_types:
        comp_sku = f"{c_type[:3].upper()}-{random.randint(100, 999)}"
        # Scenario Logic: Some components are missing to trigger the Procurement Agent
        comp_stock = random.randint(10, 50) if random.random() > 0.15 else 0

        bom_rows.append(
            {
                "Assembly_SKU": bike["SKU"],
                "Component_SKU": comp_sku,
                "Component_Name": f"{c_type} Specialist",
                "Qty_Needed": 1 if c_type != "Tires" else 2,
                "Component_Stock": comp_stock,
            }
        )

# Write to CSVs
with open(OUTPUT_ASSEMBLY, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["SKU", "Name", "Stock"])
    writer.writeheader()
    writer.writerows(assemblies)

with open(OUTPUT_BOM, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Assembly_SKU",
            "Component_SKU",
            "Component_Name",
            "Qty_Needed",
            "Component_Stock",
        ],
    )
    writer.writeheader()
    writer.writerows(bom_rows)

print(f"Generated {NUM_ASSEMBLIES} assemblies and {len(bom_rows)} BOM rows.")
