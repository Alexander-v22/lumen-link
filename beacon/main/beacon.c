#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"

#define BEACON_GPIO     GPIO_NUM_18
#define BLINK_HZ        5
#define HALF_PERIOD_MS  (1000 / (2 * BLINK_HZ))   // 100 ms on, 100 ms off

void app_main(void)
{
    gpio_reset_pin(BEACON_GPIO);
    gpio_set_direction(BEACON_GPIO, GPIO_MODE_OUTPUT);

    TickType_t last_wake = xTaskGetTickCount();
    int level = 0;

    while (1) {
        level = !level;
        gpio_set_level(BEACON_GPIO, level);
        xTaskDelayUntil(&last_wake, pdMS_TO_TICKS(HALF_PERIOD_MS));
    }
}