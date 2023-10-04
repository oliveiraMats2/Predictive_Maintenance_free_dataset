import datetime
import time
import threading
import socket_tools as ws
from inference_multi_variable import multivariate_main


class SchedulerTask:
    def day_task(self):
        now = datetime.datetime.now()
        print(f"Result time execute: {now_}")

    def task_exec(self, init_data):
        multivariate_main(init_data)

    def call_scheduler(self, scheduler):
        while True:
            now = datetime.datetime.now()
            now_hour = now.hour
            now_min = now.minute
            now_year = now.year
            now_month = now.month
            now_day = now.day

            if (
                now_year == scheduler.year
                and now_month == scheduler.month
                and now_day == scheduler.day
                and now_hour == scheduler.hour
                and now_min == scheduler.minute
            ):
                self.task_exec(scheduler)
                break

            time.sleep(1)

    @staticmethod
    def increment_actual_data(data, dias=1, mins=0):
        nova_data = data + datetime.timedelta(days=dias, minutes=mins)
        return nova_data


def keep_server():
    print('System has started websocket')
    while True:
        ws.socket_receive(ws.check_ip()[0])
        print('System wait some time before restart websocket')
        time.sleep(5)
    return


if __name__ == "__main__":
    # schedule_multi_variable = datetime.datetime(year=2023,
    #                                             month=8,
    #                                             day=1,
    #                                             hour=15,
    #                                             minute=23)

    now_ = datetime.datetime.now()

    scheduler_task = SchedulerTask()

    print(f"Result time execute: {now_}")
    multivariate_main(now_)

    t = threading.Thread(target=keep_server)
    t.start()

    while True:
        now_ = scheduler_task.increment_actual_data(now_, dias=0, mins=60)
        scheduler_task.call_scheduler(now_)
