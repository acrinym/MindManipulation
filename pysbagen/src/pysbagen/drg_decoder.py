import base64

S = [
    22,  213, 140,  67, 234,  48, 108, 225,   6, 101, 194,  50,  44,
    247,  58, 145,  20,  80, 241,  60, 127, 154, 125,  33,  45, 166,
    245,  84,  28, 110, 220,  56, 195, 181, 238, 109,  69, 216,  31,
    162,  61, 183,  74,  71, 129, 148, 170, 111, 137, 164, 179, 178,
    9,    41, 160, 219,  77,  93,  97, 143,  14, 158, 118, 152,   0,
    221, 192, 116,  86,  65,  55, 173, 217,  32, 227, 119, 102, 115,
    254, 132,  95,  23,  49,  73, 211, 142,  66,  59,  85, 252, 138,
    212, 243,  38, 134, 165, 184,  13, 209, 124, 197, 141, 114,  43,
    92,  133, 175, 205, 128,  68,  91, 104,  64, 126,  39,  40,  46,
    72,  139, 232, 182,   2, 131, 201, 188, 112, 200,  78, 159, 113,
    237,  99, 249,  90,   7,  47, 122,  36,  76, 117, 222, 149,  96,
    82,  100, 208, 151, 198, 228,  94,  87, 190,  42, 246,  10, 169,
    171, 120,  51, 236, 255, 215, 191, 223,  54, 103,  89, 135,  57,
    98,  176, 161,  24, 235,  26,   3, 250, 233, 121,  79, 207, 242,
    224,  11, 123, 193, 155, 157, 218, 186, 244,  75, 167,  63, 206,
    81,   29, 150, 229,   4,  15, 230,  37, 185,   1, 203,  35,  16,
    136, 204, 144, 253, 214, 168,  27, 189, 105, 231, 177,  18,  25,
    52,   70,  88, 196, 210, 163, 239, 156,  19,  34,  17, 202,  30,
    21,   62, 147, 174, 240, 130,   8, 180, 106, 172,  83,  12, 146,
    251, 226,  53, 153, 107, 199, 248, 187,   5
]

def rc4_decrypt(data):
    """Decrypts data using RC4 with the hardcoded key."""
    s = list(S) # Make a copy of the key
    i = 0
    j = 0
    out = []
    for char in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(char ^ s[(s[i] + s[j]) % 256])
    return bytes(out)

def parse_drg_file(filepath):
    """Parses a .drg file and returns a list of Base64 decoded elements."""
    with open(filepath, 'r') as f:
        content = f.read()

    # The first element is the header, which is not separated by @
    # but is terminated by a newline.
    parts = content.split('@')
    header_part = parts[0].splitlines()[0]

    # The rest of the parts are the other elements
    elements = [header_part] + parts[1:]

    decoded_elements = []
    for element in elements:
        # The C code seems to ignore newlines in the base64 data
        element = element.replace('\n', '').replace('\r', '')
        try:
            decoded = base64.b64decode(element)
            decoded_elements.append(decoded)
        except base64.binascii.Error as e:
            # It's possible that some elements are not base64 encoded.
            # The C code doesn't seem to handle this, but we should.
            # For now, we'll just append the raw element.
            print(f"Warning: Could not decode element: {e}")
            decoded_elements.append(element.encode('utf-8'))

    return decoded_elements

def decode_drg_image(data):
    """Decodes the image data from a .drg file."""
    # The image data is base64 encoded twice
    try:
        return base64.b64decode(data)
    except:
        return None

def decode_drg(filepath):
    """Decodes a .drg file and returns the SBG data and image data."""
    decoded_elements = parse_drg_file(filepath)

    if len(decoded_elements) < 5:
        raise ValueError("Invalid .drg file: not enough elements.")

    # The SBG data is the 5th element
    sbg_data_encrypted = decoded_elements[4]

    # Decrypt the SBG data
    sbg_data_decrypted = rc4_decrypt(sbg_data_encrypted)

    # The image data is the 3rd element
    image_data_encrypted = decoded_elements[2]
    image_data_decrypted = rc4_decrypt(image_data_encrypted)
    image_data = decode_drg_image(image_data_decrypted)

    return sbg_data_decrypted.decode('utf-8'), image_data
