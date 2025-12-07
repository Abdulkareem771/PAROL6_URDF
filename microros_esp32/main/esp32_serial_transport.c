#include <uxr/client/transport.h>

#include <driver/uart.h>
#include <driver/gpio.h>
#include <esp_log.h>

// Hardcoded pins for UART0
#define UART_TXD  (1)
#define UART_RXD  (3)
#define UART_RTS  (UART_PIN_NO_CHANGE)
#define UART_CTS  (UART_PIN_NO_CHANGE)

// --- micro-ROS Transports ---
#define UART_BUFFER_SIZE (512)

static const char *TAG = "esp32_serial_transport";

bool esp32_serial_open(struct uxrCustomTransport * transport)
{
    if (transport == NULL || transport->args == NULL) {
        ESP_LOGE(TAG, "transport or args NULL");
        return false;
    }

    uart_port_t *uart_port = (uart_port_t *) transport->args;

    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };

    if (uart_param_config(*uart_port, &uart_config) != ESP_OK) {
        ESP_LOGE(TAG, "uart_param_config failed");
        return false;
    }
    if (uart_set_pin(*uart_port, UART_TXD, UART_RXD, UART_RTS, UART_CTS) != ESP_OK) {
        ESP_LOGE(TAG, "uart_set_pin failed");
        return false;
    }
    if (uart_driver_install(*uart_port, UART_BUFFER_SIZE * 2, 0, 0, NULL, 0) != ESP_OK) {
        ESP_LOGE(TAG, "uart_driver_install failed");
        return false;
    }

    ESP_LOGI(TAG, "UART transport opened on port %d", *uart_port);
    return true;
}

bool esp32_serial_close(struct uxrCustomTransport * transport)
{
    if (transport == NULL || transport->args == NULL) {
        return false;
    }
    uart_port_t *uart_port = (uart_port_t *) transport->args;

    return uart_driver_delete(*uart_port) == ESP_OK;
}

size_t esp32_serial_write(struct uxrCustomTransport* transport, const uint8_t * buf, size_t len, uint8_t * err)
{
    if (transport == NULL || transport->args == NULL || buf == NULL) {
        if (err) *err = 1;
        return 0;
    }

    uart_port_t *uart_port = (uart_port_t *) transport->args;
    int txBytes = uart_write_bytes(*uart_port, (const char*) buf, (size_t)len);

    if (txBytes < 0) {
        if (err) *err = 1;
        return 0;
    }

    return (size_t) txBytes;
}

size_t esp32_serial_read(struct uxrCustomTransport* transport, uint8_t* buf, size_t len, int timeout, uint8_t* err)
{
    if (transport == NULL || transport->args == NULL || buf == NULL) {
        if (err) *err = 1;
        return 0;
    }

    uart_port_t *uart_port = (uart_port_t *) transport->args;
    int rx = uart_read_bytes(*uart_port, buf, (size_t)len, timeout / portTICK_PERIOD_MS);

    if (rx < 0) {
        if (err) *err = 1;
        return 0;
    }
    return (size_t) rx;
}
