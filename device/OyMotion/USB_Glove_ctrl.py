from usb_glove_data_ubuntu import *
from ROHand import *

async def main():
    NODE_ID=2
    COM_PORT="/dev/ttyUSB3"
    client = ROHand(COM_PORT,NODE_ID)
    client.connect()
    # glove_ctrl = OGlove()
    serial_port = serial.Serial(
        port=find_comport(),  # 自动检测或默认COM1
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
    )

    print(f"Using serial port: {serial_port.name}")
    glove_ctrl = OGlove(serial=serial_port, timeout=2000)
    await glove_ctrl.calib(flag=False)
    
    # import pdb;pdb.set_trace()
    # exit(0)
    # await glove_ctrl.connect_gforce_device()
    print("finish calib\n")
    client.reset()
    print("finish reset\n")

    while not glove_ctrl.terminated:
        await glove_ctrl.get_pos()
        resp = client.set_finger_pos(ROH_FINGER_POS_TARGET0, glove_ctrl.finger_data)

    await glove_ctrl.gforce_device.stop_streaming()
    await glove_ctrl.gforce_device.disconnect()
    

if __name__ == "__main__":
    asyncio.run(main())
