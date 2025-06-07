import time
import serial
from serial.tools import list_ports
import asyncio

# Constants

MAX_PROTOCOL_DATA_SIZE = 64

# Protocol states
WAIT_ON_HEADER_0 = 0
WAIT_ON_HEADER_1 = 1
WAIT_ON_BYTE_COUNT = 2
WAIT_ON_DATA = 3
WAIT_ON_LRC = 4

# Protocol byte name
DATA_CNT_BYTE_NUM = 0
DATA_START_BYTE_NUM = 1


# OHand bus context
class OGlove:
    def __init__(self, serial, timeout):
        """
        Initialize OGlove.

        Parameters
        ----------
        serial : str
            Path to serial port
        timeout : int
            Timeout in in milliseconds
        """
        self.serial_port = serial
        self.timeout = timeout
        self.is_whole_packet = False
        self.decode_state = WAIT_ON_HEADER_0
        self.packet_data = bytearray(MAX_PROTOCOL_DATA_SIZE + 2)  # Including byte_cnt, data[], lrc
        self.send_buf = bytearray(MAX_PROTOCOL_DATA_SIZE + 4)  # Including header0, header1, nb_data, lrc
        self.byte_count = 0
        self.emg_min=[0 for _ in range(6)]
        self.emg_max=[0 for _ in range(6)]
        self.finger_data=[0 for _ in range(6)]
        self.NUM_FINGERS=5
        self.terminated = False

    def calc_lrc(ctx, lrcBytes, lrcByteCount):
        """
        Calculate the LRC for a given sequence of bytes
        :param lrcBytes: sequence of bytes to calculate LRC over
        :param lrcByteCount: number of bytes in the sequence
        :return: calculated LRC value
        """
        lrc = 0
        for i in range(lrcByteCount):
            lrc ^= lrcBytes[i]
        return lrc

    def on_data(self, data):
        """
        Called when a new byte is received from the serial port. This function implements
        a state machine to decode the packet. If a whole packet is received, is_whole_packet
        is set to 1 and the packet is stored in packet_data.

        Args:
            data (int): The newly received byte

        Returns:
            None
        """
        if self is None:
            return

        if self.is_whole_packet:
            return  # Old packet is not processed, ignore

        # State machine implementation
        if self.decode_state == WAIT_ON_HEADER_0:
            if data == 0x55:
                self.decode_state = WAIT_ON_HEADER_1

        elif self.decode_state == WAIT_ON_HEADER_1:
            self.decode_state = WAIT_ON_BYTE_COUNT if data == 0xAA else WAIT_ON_HEADER_0

        elif self.decode_state == WAIT_ON_BYTE_COUNT:
            self.packet_data[DATA_CNT_BYTE_NUM] = data
            self.byte_count = data

            if self.byte_count > MAX_PROTOCOL_DATA_SIZE:
                self.decode_state = WAIT_ON_HEADER_0
            elif self.byte_count > 0:
                self.decode_state = WAIT_ON_DATA
            else:
                self.decode_state = WAIT_ON_LRC

        elif self.decode_state == WAIT_ON_DATA:
            self.packet_data[DATA_START_BYTE_NUM + self.packet_data[DATA_CNT_BYTE_NUM] - self.byte_count] = data
            self.byte_count -= 1

            if self.byte_count == 0:
                self.decode_state = WAIT_ON_LRC

        elif self.decode_state == WAIT_ON_LRC:
            self.packet_data[DATA_START_BYTE_NUM + self.packet_data[DATA_CNT_BYTE_NUM]] = data
            self.is_whole_packet = True
            self.decode_state = WAIT_ON_HEADER_0

        else:
            self.decode_state = WAIT_ON_HEADER_0

    def get_data(self, resp_bytes) -> bool:
        """
        Retrieve a complete packet from the serial port and validate it.

        Args:
            resp_bytes (bytearray): A bytearray to store the response data.

        Returns:
            bool: True if a valid packet is received, False otherwise.
        """
        # Check if self or self.serial_port is None
        if self is None or self.serial_port is None:
            return False

        # 记录开始等待的时间
        wait_start = time.time()
        # 计算等待超时时间
        wait_timeout = wait_start + self.timeout / 1000

        # 循环等待完整的数据包
        while not self.is_whole_packet:
            # time.sleep(0.001)

            # print(f"in_waiting: {self.serial_port.in_waiting}")

            # 如果串口有数据可读
            while self.serial_port.in_waiting > 0:
                # 读取串口数据
                data_bytes = self.serial_port.read(self.serial_port.in_waiting)
                # print("data_bytes: ", len(data_bytes))

                # 遍历读取到的数据
                for ch in data_bytes:
                    # print(f"data: {hex(ch)}")
                    # 处理数据
                    self.on_data(ch)
                # 如果已经读取到完整的数据包，跳出循环
                if self.is_whole_packet:
                    break

            # 如果还没有读取到完整的数据包，并且已经超时，跳出循环
            if (not self.is_whole_packet) and (wait_timeout < time.time()):
                # print(f"wait time out: {wait_timeout}, now: {time.time()}")
                # 重置解码状态
                self.decode_state = WAIT_ON_HEADER_0
                return False

        # Validate LRC
        lrc = self.calc_lrc(self.packet_data, self.packet_data[DATA_CNT_BYTE_NUM] + 1)
        if lrc != self.packet_data[DATA_START_BYTE_NUM + self.packet_data[DATA_CNT_BYTE_NUM]]:
            self.is_whole_packet = False
            return False

        # Copy response data
        if resp_bytes is not None:
            packet_byte_count = self.packet_data[DATA_CNT_BYTE_NUM]
            resp_bytes.clear()
            resp_data = self.packet_data[DATA_START_BYTE_NUM : DATA_START_BYTE_NUM + packet_byte_count]
            for v in resp_data:
                resp_bytes.append(v)

        self.is_whole_packet = False
        return True
    
    @staticmethod
    def clamp(n, smallest, largest):
        return max(smallest, min(n, largest))
    
    @staticmethod
    def _signal_handler(self):
        print("You pressed ctrl-c, exit")
        self.terminated = True
        
    @staticmethod
    def interpolate(n, from_min, from_max, to_min, to_max):
        return (n - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
    
    async def get_pos(self):
        glove_data = bytearray()
        
        if self.get_data(glove_data):
            finger_data = []
            for i in range(int(len(glove_data) / 2)):
                finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
            for j in range(self.NUM_FINGERS+1):
                deadzone=3500 if j in [1,2,3,4] else 7000
                raw_value=round(self.interpolate(finger_data[j], self.emg_min[j], self.emg_max[j], 65535, 0))
                if abs(raw_value-self.finger_data[j]) < deadzone:
                    continue
                else:
                    self.finger_data[j] = self.clamp(raw_value, 0, 65535)
            print(self.finger_data)    
        
    async def calib(self,flag=True):    
        if not flag:
            self.emg_max=[1009, 414, 1066, 1272, 1140, 684]
            self.emg_min=[605, 266, 692, 583, 596, 466]
            return
        input("thumb max")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_max[0]=round((self.emg_max[0]+finger_data[0])/2)
        print(self.emg_max, finger_data)
        input("thumb min")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_min[0]=round((self.emg_min[0]+finger_data[0])/2)
        print(self.emg_min, finger_data)
        input("second finger max")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_max[1]=round((self.emg_max[1]+finger_data[1])/2)
        print(self.emg_max, finger_data)
        input("second finger min")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_min[1]=round((self.emg_min[1]+finger_data[1])/2)
        print(self.emg_min, finger_data)
        input("third finger max")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_max[2]=round((self.emg_max[2]+finger_data[2])/2)
        print(self.emg_max, finger_data)
        input("third finger min")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_min[2]=round((self.emg_min[2]+finger_data[2])/2)
        print(self.emg_min, finger_data)
        input("fourth finger max")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_max[3]=round((self.emg_max[3]+finger_data[3])/2)
        print(self.emg_max, finger_data)
        input("fourth finger min")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_min[3]=round((self.emg_min[3]+finger_data[3])/2)
        print(self.emg_min, finger_data)
        input("fifth finger max")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_max[4]=round((self.emg_max[4]+finger_data[4])/2)
        print(self.emg_max, finger_data)
        input("fifth finger min")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_min[4]=round((self.emg_min[4]+finger_data[4])/2)
        print(self.emg_min, finger_data)
        input("thumb rotation max")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_max[5]=round((self.emg_max[5]+finger_data[5])/2)
        print(self.emg_max, finger_data)
        input("thumb rotation min")
        for _ in range(256):
            glove_data = bytearray()
            if self.get_data(glove_data):
                finger_data = []
                for i in range(int(len(glove_data) / 2)):
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
                self.emg_min[5]=round((self.emg_min[5]+finger_data[5])/2)
        print(self.emg_min, finger_data)
        # for _ in range(256):
        #     glove_data = bytearray()
        #     if self.get_data(glove_data):
        #         finger_data = []
        #         for i in range(int(len(glove_data) / 2)):
        #             finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
        #         for i in range(self.NUM_FINGERS+1): # this gives the largest value for thumb rotation
        #             self.emg_max[i]=round((self.emg_max[i]+finger_data[i])/2)
        # print(self.emg_max, finger_data)
        # input("Please make a fist")
        # for _ in range(256):
        #     glove_data = bytearray()
        #     if self.get_data(glove_data):
        #         finger_data = []
        #         for i in range(int(len(glove_data) / 2)):
        #             finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
        #         for i in range(self.NUM_FINGERS):
        #             self.emg_min[i]=round((self.emg_min[i]+finger_data[i])/2)
        # print(self.emg_min, finger_data)
        # input("Please rotate your thumb root to maximum angle")
        # for _ in range(256):
        #     glove_data = bytearray()
        #     if self.get_data(glove_data):
        #         finger_data = []
        #         for i in range(int(len(glove_data) / 2)):
        #             finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))
        #         self.emg_min[5]=round((self.emg_min[5]+finger_data[5])/2)
        print(self.emg_min,self.emg_max)
        import pdb;pdb.set_trace()
        print(self.emg_min,self.emg_max)
def find_comport():
    """自动查找可用串口"""
    ports = list_ports.comports()
    for port in ports:
        # if "ttyUSB" in port.device or "ttyACM" in port.device:
        if "ttyACM" in port.device:
            return port.device
    return None


async def main():
    # 配置串口参数（根据实际设备修改）
    serial_port = serial.Serial(
        port=find_comport(),  # 自动检测或默认COM1
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
    )

    print(f"Using serial port: {serial_port.name}")
    oglove = OGlove(serial=serial_port, timeout=2000)
    await oglove.calib()
    exit(0)
    try:
        glove_data = bytearray()
        while True:
            # 读取串口数据
            if oglove.get_data(glove_data):
                finger_data = []
                # print("Received data:", glove_data.hex(" ", 1))

                # 处理数据
                for i in range(int(len(glove_data) / 2)):
                    # 每两个字节为一个数据
                    finger_data.append((glove_data[i * 2]) | (glove_data[i * 2 + 1] << 8))

                print("大拇指弯曲:{0:5},  食指弯曲:{1:5},  中指弯曲:{2:5},  无名指弯曲:{3:5},  小拇指弯曲:{4:5},  大拇指旋转:{5:5}"
                    .format(finger_data[0], finger_data[1], finger_data[2], finger_data[3], finger_data[4], finger_data[5]))
    except KeyboardInterrupt:
        print("用户终止程序")
    finally:
        serial_port.close()
        print("串口已关闭")


if __name__ == "__main__":
    asyncio.run(main())