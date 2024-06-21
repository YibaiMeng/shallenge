#ifndef SHA256_HASH_H
#define SHA256_HASH_H
#include <stdint.h>
#include <iostream>

struct sha256_hash
{
    uint32_t hash[8];
    friend std::ostream &operator<<(std::ostream &os, const sha256_hash &obj);
};
std::ostream &operator<<(std::ostream &os, const sha256_hash &obj)
{
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "%08x %08x %08x %08x %08x %08x %08x %08x", obj.hash[0], obj.hash[1], obj.hash[2], obj.hash[3], obj.hash[4], obj.hash[5], obj.hash[6], obj.hash[7]);

    os << std::string(buffer);
    return os;
}

#endif // SHA256_HASH_H