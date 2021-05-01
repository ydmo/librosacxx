#ifndef TESTSHELPER_H
#define TESTSHELPER_H

#include <string>

inline bool is_base64(unsigned char c) {
        return (isalnum(c) || (c == '+') || (c == '/'));
}

inline std::string Base64Decode(const std::string& i_encoded_string) {
    const std::string StrBase64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    auto in_len = i_encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;
    while (in_len-- && ( i_encoded_string[in_] != '=') && is_base64(i_encoded_string[in_])) {
        char_array_4[i++] = i_encoded_string[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++) {
                char_array_4[i] = StrBase64Chars.find(char_array_4[i]);
            }
            char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];
            for (i = 0; (i < 3); i++) {
                ret += char_array_3[i];
            }
            i = 0;
        }
    }
    if (i) {
        for (j = 0; j < i; j++) {
            char_array_4[j] = StrBase64Chars.find(char_array_4[j]);
        }
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        for (j = 0; (j < i - 1); j++) {
            ret += char_array_3[j];
        }
    }
    return ret;
}

#endif // TESTSHELPER_H
