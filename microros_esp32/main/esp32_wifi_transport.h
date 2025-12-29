#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <uxr/client/profile/transport/custom/custom_transport.h>

bool esp32_wifi_open(struct uxrCustomTransport * transport);
bool esp32_wifi_close(struct uxrCustomTransport * transport);
size_t esp32_wifi_write(struct uxrCustomTransport * transport,
                        const uint8_t * buf,
                        size_t len,
                        uint8_t * errcode);
size_t esp32_wifi_read(struct uxrCustomTransport * transport,
                       uint8_t * buf,
                       size_t len,
                       int timeout,
                       uint8_t * errcode);

