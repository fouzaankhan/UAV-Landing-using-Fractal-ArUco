import json
import os

print("🔧 Creating proper AirSim settings...")

# Create the directory
os.makedirs("C:/Users/khanf/Documents/AirSim", exist_ok=True)

# Settings that work with SimpleFlight
settings = {
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "Vehicles": {
        "Drone1": {
            "VehicleType": "SimpleFlight",
            "AutoCreate": True,
            "DefaultVehicleState": "Armed",
            "AllowAPIAlways": True
        }
    }
}

# Save the settings
settings_path = "C:/Users/khanf/Documents/AirSim/settings.json"
with open(settings_path, "w") as f:
    json.dump(settings, f, indent=4)

print(f"✅ Settings created: {settings_path}")
print("✅ Key change: DefaultVehicleState = Armed")
print("\nNext steps:")
print("1. Close AirSimNH.exe completely")
print("2. Wait 10 seconds")
print("3. Start AirSimNH.exe again")
print("4. Look for spinning propellers when it loads")