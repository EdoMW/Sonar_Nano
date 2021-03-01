# Echo client program
import socket
import time
import struct
import codecs


class ReadFromRobot:
    def __init__(self):
        self.HOST = "132.72.96.97"  # The remote host
        self.PORT_30003 = 30003
        self.s = 0
        self.packet_1, self.packet_2, self.packet_3, self.packet_4, self.packet_5, self.packet_6, self.packet_7, \
            self.packet_8, self.packet_9, self.packet_10 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_11, self.packet_12, self.packet_13, self.packet_14, self.packet_15, self.packet_16, \
            self.packet_17, self.packet_18, self.packet_19, self.packet_20 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_21, self.packet_22, self.packet_23, self.packet_24, self.packet_25, self.packet_26, \
            self.packet_27, self.packet_28, self.packet_29, self.packet_30 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_31, self.packet_32, self.packet_33, self.packet_34, self.packet_35, self.packet_36, \
            self.packet_37, self.packet_38, self.packet_39, self.packet_40 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_41, self.packet_42, self.packet_43, self.packet_44, self.packet_45, self.packet_46, \
            self.packet_47, self.packet_48, self.packet_49, self.packet_50 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_51, self.packet_52, self.packet_53, self.packet_54, self.packet_55, self.packet_56, \
            self.packet_57, self.packet_58, self.packet_59, self.packet_60 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_61, self.packet_62, self.packet_63, self.packet_64, self.packet_65, self.packet_66, \
            self.packet_67, self.packet_68, self.packet_69, self.packet_70 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.packet_71, self.packet_72, self.packet_73 = 0, 0, 0
        self.x, self.y, self.z, self.Rx, self.Ry, self.Rz = 0, 0, 0, 0, 0, 0

        #self.connect_to_robot()

    def __del__(self):
        try:
            print("disconnecting to UR5.. \nclose read from socket", self.s)
            self.s.close()
            time.sleep(0.1)
        except socket.error as socketError:
            print("Error: ", socketError)

    def connect_to_robot(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("connecting to UR5.. \nopen read from socket", self.s)
            self.s.settimeout(10)
            self.s.connect((self.HOST, self.PORT_30003))
            time.sleep(0.1)
        except socket.error as socketError:
            print("Error: ", socketError)

    def disconnect_to_robot(self):
        try:
            print("disconnecting to UR5.. \nclose read from socket", self.s)
            self.s.detach()
            time.sleep(0.1)
        except socket.error as socketError:
            print("Error: ", socketError)

    def read_from_robot(self):
        # 3-8 Returns the desired angular position of all joints
        self.packet_1 = self.s.recv(4)
        self.packet_2 = self.s.recv(8)
        self.packet_3 = self.s.recv(8)
        self.packet_4 = self.s.recv(8)
        self.packet_5 = self.s.recv(8)
        self.packet_6 = self.s.recv(8)
        self.packet_7 = self.s.recv(8)
        self.packet_8 = self.s.recv(8)
        # 9-14 Returns the desired angular velocities of all joints
        self.packet_9 = self.s.recv(8)
        self.packet_10 = self.s.recv(8)
        self.packet_11 = self.s.recv(8)
        self.packet_12 = self.s.recv(8)
        self.packet_13 = self.s.recv(8)
        self.packet_14 = self.s.recv(8)
        # 15-20 Returns the desired angular accelerations of all joints
        self.packet_15 = self.s.recv(8)
        self.packet_16 = self.s.recv(8)
        self.packet_17 = self.s.recv(8)
        self.packet_18 = self.s.recv(8)
        self.packet_19 = self.s.recv(8)
        self.packet_20 = self.s.recv(8)
        # 21-26 Returns the desired Currents of all joints -- not used
        self.packet_21 = self.s.recv(8)
        self.packet_22 = self.s.recv(8)
        self.packet_23 = self.s.recv(8)
        self.packet_24 = self.s.recv(8)
        self.packet_25 = self.s.recv(8)
        self.packet_26 = self.s.recv(8)
        # 27-32 Returns the desired Torques of all joints
        self.packet_27 = self.s.recv(8)
        self.packet_28 = self.s.recv(8)
        self.packet_29 = self.s.recv(8)
        self.packet_30 = self.s.recv(8)
        self.packet_31 = self.s.recv(8)
        self.packet_32 = self.s.recv(8)
        # 33 Returns the same information as packages 3-8 -- not used again
        self.packet_33 = self.s.recv(48)
        # 34 Returns the same information as packages 9-14 -- not used again
        self.packet_34 = self.s.recv(48)
        # 35 Returns the same information as packages 15-20 -- not used again
        self.packet_35 = self.s.recv(48)
        # 36 Returns the same information as packages 21-26 -- not used
        self.packet_36 = self.s.recv(48)
        # 37-42 Returns actual cartesian coordinates of the tool
        self.packet_37 = self.s.recv(8)
        self.packet_38 = self.s.recv(8)
        self.packet_39 = self.s.recv(8)
        self.packet_40 = self.s.recv(8)
        self.packet_41 = self.s.recv(8)
        self.packet_42 = self.s.recv(8)
        # 43- 48 Returns actual speed of the tool given in cartesian coordinates.The first three values are the
        # cartesian speeds along x, y, z. and the last three define the current rotation axis, rx, ry, rz, and the
        # length | rz, ry, rz | defines the angular velocity in radians / s.
        self.packet_43 = self.s.recv(8)
        self.packet_44 = self.s.recv(8)
        self.packet_45 = self.s.recv(8)
        self.packet_46 = self.s.recv(8)
        self.packet_47 = self.s.recv(8)
        self.packet_48 = self.s.recv(8)
        # 49 Returns the generalized force on the tcp -- not used
        self.packet_49 = self.s.recv(48)
        # 50 Returns Target Cartesian coordinates of the tool: (x,y,z,rx,ry,rz), where rx, ry and rz is a rotation
        # vector representation of the tool orientation --not used
        self.packet_50 = self.s.recv(48)
        # 51 Returns Target speed of the tool given in Cartesian coordinates -- not used
        self.packet_51 = self.s.recv(48)
        # 52 Returns Current state of the digital inputs. NOTE: these are bits encoded as int64_t,
        # e.g. a value of 5 corresponds to bit 0 and bit 2 set high
        self.packet_52 = self.s.recv(8)
        # 53 Returns Temperature of each joint in degrees celsius -- not used
        self.packet_53 = self.s.recv(48)
        # 54 Returns Controller realtime thread execution time
        self.packet_54 = self.s.recv(8)
        # 55 Returns Test Value - A value used by Universal Robots software only -- not used
        self.packet_55 = self.s.recv(8)
        # 56 Returns robot mode - see DataStreamFromURController in the Excel file for more information -- not used
        self.packet_56 = self.s.recv(8)
        # 57 Returns Joint control mode -see DataStreamFromURController in the Excel file for more information -- not
        # used
        self.packet_57 = self.s.recv(48)
        # 58 Returns Safety mode - see	DataStreamFromURController in the Excel file for more information -- not used
        self.packet_58 = self.s.recv(8)
        # 59 is Used by Universal Robots software only
        self.packet_59 = self.s.recv(48)
        # 60-62 Returns the current reading of the tool accelerometer as a three-dimensional vector.
        # The accelerometer axes are aligned with the tool coordinates, and pointing an axis upwards
        # results in a positive reading.
        self.packet_60 = self.s.recv(8)
        self.packet_61 = self.s.recv(8)
        self.packet_62 = self.s.recv(8)
        # 63 is Used by Universal Robots software only
        self.packet_63 = self.s.recv(48)
        # 64 Returns Speed scaling of the trajectory limiter -- not used
        self.packet_64 = self.s.recv(8)
        # 65 Returns Norm of Cartesian linear momentum --not used
        self.packet_65 = self.s.recv(8)
        # 66&67 are Used by Universal Robots software only
        self.packet_66 = self.s.recv(8)
        self.packet_67 = self.s.recv(8)
        # 68 Returns the main voltage -- not used
        self.packet_68 = self.s.recv(8)
        # 69 Returns the robot voltage (48V) -- not used
        self.packet_69 = self.s.recv(8)
        # 70 Returns the robot current --not used
        self.package_70 = self.s.recv(8)
        # 71 Returns Actual joint voltages -- not used
        self.package_71 = self.s.recv(48)
        # 72 Returns digital outputs --not used
        self.package_72 = self.s.recv(8)
        # 73 Returns program state --not used
        self.package_73 = self.s.recv(8)

    def iget_act_joint_pos(self):
        packet_3 = self.packet_3.encode("hex")  # convert the data from \x hex notation to plain hex
        BaseInString = str(packet_3)
        base = struct.unpack('!d', packet_3.decode('hex'))[0]
        # print("Base in Deg = ", base * (180 / pi))
        packet_4 = self.packet_4.encode("hex")  # convert the data from \x hex notation to plain hex
        ShoulderInString = str(packet_4)
        shoulder = struct.unpack('!d', packet_4.decode('hex'))[0]
        # print("Shoulder in Deg = ", shoulder * (180 / pi))
        packet_5 = self.packet_5.encode("hex")  # convert the data from \x hex notation to plain hex
        ElbowInString = str(packet_5)
        elbow = struct.unpack('!d', packet_5.decode('hex'))[0]
        # print("Elbow in Deg = ", elbow * (180 / pi))
        packet_6 = self.packet_6.encode("hex")  # convert the data from \x hex notation to plain hex
        Wrist1InString = str(packet_6)
        wrist1 = struct.unpack('!d', packet_6.decode('hex'))[0]
        # print("Wrist1 in Deg = ", wrist1 * (180 / pi))
        packet_7 = self.packet_7.encode("hex")  # convert the data from \x hex notation to plain hex
        Wrist2InString = str(packet_7)
        wrist2 = struct.unpack('!d', packet_7.decode('hex'))[0]
        # print("Wrist2 in Deg = ", wrist2 * (180 / pi))
        packet_8 = self.packet_8.encode("hex")  # convert the data from \x hex notation to plain hex
        Wrist3InString = str(packet_8)
        wrist3 = struct.unpack('!d', packet_8.decode('hex'))[0]
        # print("Wrist3 in Deg = ", wrist3 * (180 / pi))

    def iget_act_joint_vel(self):
        packet_9 = codecs.encode(self.packet_9, 'hex_codec')  # convert the data from \x hex notation to plain hex
        BaseVInString = str(packet_9)
        baseV = struct.unpack('!d', self.packet_9)[0]
        # print("Base Velocity in rad\sec = ", baseV)
        packet_10 = codecs.encode(self.packet_10, 'hex_codec')  # convert the data from \x hex notation to plain hex
        ShoulderVInString = str(packet_10)
        shoulderV = struct.unpack('!d', self.packet_10)[0]
        # print("Shoulder Velocity in rad\sec = ", shoulderV)
        packet_11 = codecs.encode(self.packet_11, 'hex_codec')  # convert the data from \x hex notation to plain hex
        ElbowVInString = str(packet_11)
        elbowV = struct.unpack('!d', self.packet_11)[0]
        # print("Elbow Velocity in rad\sec = ", elbowV)
        packet_12 = codecs.encode(self.packet_12, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist1InStringV = str(packet_12)
        wrist1V = struct.unpack('!d', self.packet_12)[0]
        # print("Wrist1 Velocity in rad\sec = ", wrist1V)
        packet_13 = codecs.encode(self.packet_13, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist2InStringV = str(packet_13)
        wrist2V = struct.unpack('!d', self.packet_13)[0]
        # print("Wrist2 Velocity in rad\sec = ", wrist2V)
        packet_14 = codecs.encode(self.packet_14, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist3InStringV = str(packet_14)
        wrist3V = struct.unpack('!d', self.packet_14)[0]
        # print("Wrist3 Velocity in rad\sec= ", wrist3V)

        return baseV, shoulderV, elbowV, wrist1V, wrist2V, wrist3V

    def iget_act_joint_a(self):
        packet_15 = codecs.encode(self.packet_15, 'hex_codec')  # convert the data from \x hex notation to plain hex
        BaseAInString = str(packet_15)
        baseA = struct.unpack('!d', self.packet_15)[0]
        # print("Base Accelerations =", baseA)
        packet_16 = codecs.encode(self.packet_16, 'hex_codec')  # convert the data from \x hex notation to plain hex
        ShoulderAInString = str(packet_16)
        shoulderA = struct.unpack('!d', self.packet_16)[0]
        # print("Shoulder Acceleration = ", shoulderA)
        packet_17 = codecs.encode(self.packet_17, 'hex_codec')  # convert the data from \x hex notation to plain hex
        ElbowAInString = str(packet_17)
        elbowA = struct.unpack('!d', self.packet_17)[0]
        # print("Elbow Acceleration = ", elbowA)
        packet_18 = codecs.encode(self.packet_18, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist1AInString = str(packet_18)
        wrist1A = struct.unpack('!d', self.packet_18)[0]
        # print("Wrist1 Acceleration = ", wrist1A)
        packet_19 = codecs.encode(self.packet_19, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist2AInString = str(packet_19)
        wrist2V = struct.unpack('!d', self.packet_19)[0]
        # print("Wrist2 Acceleration = ", wrist2V)
        packet_20 = codecs.encode(self.packet_20, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist3AInString = str(packet_20)
        wrist3A = struct.unpack('!d', self.packet_20)[0]
        # print("Wrist3 Acceleration= ", wrist3A)

    def iget_act_torques(self):
        packet_27 = codecs.encode(self.packet_27, 'hex_codec')  # convert the data from \x hex notation to plain hex
        BaseTInString = str(packet_27)
        baseT = struct.unpack('!d', self.packet_27)[0]
        # print("Base Torque in NM =", baseT)
        packet_28 = codecs.encode(self.packet_28, 'hex_codec')  # convert the data from \x hex notation to plain hex
        ShoulderTInString = str(packet_28)
        shoulderT = struct.unpack('!d', self.packet_28)[0]
        # print("Shoulder Torque in NM = ", shoulderT)
        packet_29 = codecs.encode(self.packet_29, 'hex_codec')  # convert the data from \x hex notation to plain hex
        ElbowTInString = str(packet_29)
        elbowT = struct.unpack('!d', self.packet_29)[0]
        # print("Elbow Torque in NM = ", elbowT)
        packet_30 = codecs.encode(self.packet_30, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist1TInString = str(packet_30)
        wrist1T = struct.unpack('!d', self.packet_30)[0]
        # print("Wrist1 Torque in NM = ", wrist1T)
        packet_31 = codecs.encode(self.packet_31, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist2InStringT = str(packet_31)
        wrist2T = struct.unpack('!d', self.packet_31)[0]
        # print("Wrist2 Torque in NM = ", wrist2T)
        packet_32 = codecs.encode(self.packet_32, 'hex_codec')  # convert the data from \x hex notation to plain hex
        Wrist3TInString = str(packet_32)
        wrist3T = struct.unpack('!d', self.packet_32)[0]
        # print("Wrist3 Torque in NM= ", wrist3T)

    def iget_tcp_position(self):  # IN BASE COORDINATES :
        # packet_1 = codecs.encode(b'self.packet_1', 'hex_codec')
        # packet_1 = codecs.encode(self.packet_1, 'hex_codec')
        # print(packet_1)

        packet_37 = codecs.encode(self.packet_37, 'hex_codec')
        xStr = packet_37.decode('utf-8')
        # print(xStr)
        self.x = struct.unpack('!d', self.packet_37)[0]
        # print(x)

        packet_38 = codecs.encode(self.packet_38, 'hex_codec')
        # print(packet_38)
        yStr = packet_38.decode('utf-8')
        self.y = struct.unpack('!d', self.packet_38)[0]
        # print(y)

        packet_39 = codecs.encode(self.packet_39, 'hex_codec')
        # print(packet_39)
        zStr = packet_39.decode('utf-8')
        self.z = struct.unpack('!d', self.packet_39)[0]
        # print(z)

        packet_40 = codecs.encode(self.packet_40, 'hex_codec')
        # print(packet_40)
        RxStr = packet_40.decode('utf-8')
        self.Rx = struct.unpack('!d', self.packet_40)[0]
        # print(Rx)

        packet_41 = codecs.encode(self.packet_41, 'hex_codec')
        # print(packet_41)
        RyStr = packet_41.decode('utf-8')
        self.Ry = struct.unpack('!d', self.packet_41)[0]
        # print(Ry)

        packet_42 = codecs.encode(self.packet_42, 'hex_codec')
        # print(packet_42)
        RzStr = packet_42.decode('utf-8')
        self.Rz = struct.unpack('!d', self.packet_42)[0]
        # print(Rz)

    def iget_tcp_velocities(self):
        packet_43 = codecs.encode(self.packet_43, 'hex_codec')  # convert the data from \x hex notation to plain hex
        xV = struct.unpack('!d', self.packet_43)[0]
        xVInString = str(packet_43)
        # print("X Velocity  = ", xV)
        packet_44 = codecs.encode(self.packet_44, 'hex_codec')  # convert the data from \x hex notation to plain hex
        yVinString = str(packet_44)
        yV = struct.unpack('!d', self.packet_44)[0]
        # print("Y Velocity = ", yV)
        packet_45 = codecs.encode(self.packet_45, 'hex_codec')  # convert the data from \x hex notation to plain hex
        zVInString = str(packet_45)
        zV = struct.unpack('!d', self.packet_45)[0]
        # print("Z Velocity = ", zV)
        packet_46 = codecs.encode(self.packet_46, 'hex_codec')  # convert the data from \x hex notation to plain hex
        RxVInString = str(packet_46)
        RxV = struct.unpack('!d', self.packet_46)[0]
        # print("Rx Velocity = ", RxV)
        packet_47 = codecs.encode(self.packet_47, 'hex_codec')  # convert the data from \x hex notation to plain hex
        RyVInString = str(packet_47)
        RyV = struct.unpack('!d', self.packet_47)[0]
        # print("Ry Velocity = ", RyV)
        packet_48 = codecs.encode(self.packet_48, 'hex_codec')  # convert the data from \x hex notation to plain hex
        RzVInString = str(packet_48)
        RzV = struct.unpack('!d', self.packet_48)[0]
        # print("Rz Velocity = ", RzV)
        return (xV, yV, zV, RxV, RyV, RzV)

    def iget_tool_accelerometer(self):
        packet_60 = codecs.encode(self.packet_60, 'hex_codec')  # convert the data from \x hex notation to plain hex
        xacc = struct.unpack('!d', self.packet_60)[0]
        xaccInString = str(packet_60)
        # print("X tool accelerometer in m\s^2  = ", xacc)
        packet_61 = codecs.encode(self.packet_61, 'hex_codec')  # convert the data from \x hex notation to plain hex
        yaccinString = str(packet_61)
        yacc = struct.unpack('!d', self.packet_61)[0]
        # print("Y tool accelerometer in m\s^2  = ", yacc)
        packet_62 = codecs.encode(self.packet_62, 'hex_codec')  # convert the data from \x hex notation to plain hex
        zaccInString = str(packet_62)
        zacc = struct.unpack('!d', self.packet_62('hex'))[0]
        # print("Z tool accelerometer in m\s^2 = ", zacc)

    def read_tcp_pos(self):
        self.connect_to_robot()
        self.read_from_robot()
        self.iget_tcp_position()
        self.disconnect_to_robot()

        return self.x, self.y, self.z, self.Rx, self.Ry, self.Rz
