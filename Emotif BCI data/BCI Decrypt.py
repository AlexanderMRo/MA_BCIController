from Crypto.Cipher import AES
import hid
def generate_aes_key(serial_number, model):
    if not serial_number or len(serial_number) < 4:
        raise ValueError("Invalid serial number length.")

    k = []

    if model == 2:  # Epoc::Standard
        k = [serial_number[-1], '\0', serial_number[-2], 'T',
             serial_number[-3], '\x10', serial_number[-4], 'B',
             serial_number[-1], '\0', serial_number[-2], 'H',
             serial_number[-3], '\0', serial_number[-4], 'P']
    else:
        k = [serial_number[-1], serial_number[-2], serial_number[-3],
             serial_number[-4], 'A', 'B', 'C', 'D', 'E', 'F', 'G',
             'H', 'I', 'J', 'K', 'L']

    key = ''.join(k)
    return key.ljust(16, '\0')[:16]


def connect_to_device():
    print("Searching for Emotiv devices...")
    for device in hid.enumerate():
        if device['vendor_id'] == 4660:
            serial_number = device['serial_number']
            print(f"Found device with serial number: {serial_number}")

            key = generate_aes_key(serial_number, 2)
            return device['vendor_id'], device['product_id'], key

    raise RuntimeError("No Emotiv device found.")




def process_data(decrypted_data):
    channels = [int.from_bytes(decrypted_data[i:i+2], 'big')
                for i in range(0, len(decrypted_data), 2)]
    return channels

def save_data_to_csv(data, filename="eeg_data.csv"):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)


vendor_id, product_id, key = connect_to_device()
cipher = AES.new(key.encode(), AES.MODE_ECB)
raw_data = hid_device.read(32)
decrypted_data = cipher.decrypt(bytes(raw_data))