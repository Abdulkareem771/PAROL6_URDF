#include "esp32_wifi_transport.h"
#include "lwip/sockets.h"
#include "esp_log.h"

static int sock = -1;
static struct sockaddr_in agent_addr;

bool esp32_wifi_open(struct uxrCustomTransport * transport)
{
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);

    agent_addr.sin_family = AF_INET;
    agent_addr.sin_port = htons(atoi(CONFIG_MICRO_ROS_AGENT_PORT));
    agent_addr.sin_addr.s_addr = inet_addr(CONFIG_MICRO_ROS_AGENT_IP);

    return sock >= 0;
}

bool esp32_wifi_close(struct uxrCustomTransport * transport)
{
    if (sock >= 0) close(sock);
    sock = -1;
    return true;
}

size_t esp32_wifi_write(struct uxrCustomTransport * transport,
                        const uint8_t * buf,
                        size_t len,
                        uint8_t * errcode)
{
    return sendto(sock, buf, len, 0,
                  (struct sockaddr *)&agent_addr,
                  sizeof(agent_addr));
}

size_t esp32_wifi_read(struct uxrCustomTransport * transport,
                       uint8_t * buf,
                       size_t len,
                       int timeout,
                       uint8_t * errcode)
{
    struct timeval tv = {
        .tv_sec = timeout / 1000,
        .tv_usec = (timeout % 1000) * 1000
    };

    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return recvfrom(sock, buf, len, 0, NULL, NULL);
}

