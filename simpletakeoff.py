import socket
import msgpack
import struct
import time

def send_command(sock, command, params=None):
    """Send a command to AirSim via socket connection"""
    if params is None:
        params = []
    
    # Create message in AirSim format
    message = [0, 1, command, params]
    packed = msgpack.packb(message)
    length = struct.pack('>I', len(packed))
    
    # Send command
    sock.send(length + packed)
    time.sleep(0.5)  # Small delay between commands

def simple_takeoff():
    """Simple takeoff sequence for AirSim drone"""
    
    print("🚁 SIMPLE AIRSIM TAKEOFF")
    print("=" * 30)
    print("Make sure AirSimNH.exe is running and propellers are spinning!")
    
    try:
        # Connect to AirSim
        print("📡 Connecting to AirSim...")
        s = socket.socket()
        s.connect(("127.0.0.1", 41451))
        print("✅ Connected successfully!")
        
        # Reset the simulation
        print("🔄 Resetting simulation...")
        send_command(s, "reset")
        time.sleep(3)
        
        # Enable API control
        print("🎮 Enabling API control...")
        send_command(s, "enableApiControl", [True, ""])
        time.sleep(1)
        
        # Arm the drone
        print("🔧 Arming drone...")
        send_command(s, "armDisarm", [True, ""])
        time.sleep(2)
        
        # Takeoff
        print("🚀 TAKING OFF - WATCH AIRSIM WINDOW!")
        print("   Drone should rise to ~3 meters...")
        send_command(s, "takeoffAsync", [30, ""])  # 30 second timeout
        time.sleep(8)  # Wait for takeoff to complete
        
        # Hover for a moment
        print("🛑 Hovering...")
        send_command(s, "hoverAsync", [""])
        time.sleep(3)
        
        # Optional: Move up a bit more
        print("📈 Moving up to 5 meters...")
        send_command(s, "moveToZAsync", [-5, 2.0, None, ""])
        time.sleep(4)
        
        # Hover again
        print("🛑 Final hover...")
        send_command(s, "hoverAsync", [""])
        time.sleep(2)
        
        print("✅ Takeoff sequence complete!")
        print("🎉 Drone should now be hovering at 5 meters altitude")
        
        # Close connection
        s.close()
        
    except ConnectionRefusedError:
        print("❌ ERROR: Cannot connect to AirSim")
        print("   Make sure AirSimNH.exe is running!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        
    print("\n" + "="*50)
    print("WHAT YOU SHOULD SEE:")
    print("1. Drone propellers spinning faster")
    print("2. Drone lifting off the ground")
    print("3. Drone rising to about 5 meters height")
    print("4. Drone hovering stably in the air")

if __name__ == "__main__":
    # Check if user is ready
    input("Press ENTER when AirSimNH.exe is running and you can see the drone...")
    
    # Run takeoff
    simple_takeoff()
    
    # Ask for confirmation
    response = input("\nDid you see the drone take off successfully? (y/n): ")
    if response.lower() == 'y':
        print("🎉 Perfect! Your AirSim setup is working correctly!")
        print("You can now run more complex flight sequences.")
    else:
        print("🔧 If takeoff didn't work, check:")
        print("   1. AirSimNH.exe is running")
        print("   2. Propellers were spinning when AirSim started")
        print("   3. No error messages in the console")
        print("   4. Try restarting AirSim and running again")