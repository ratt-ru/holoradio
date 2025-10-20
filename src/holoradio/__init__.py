from holoscan.schedulers import EventBasedScheduler


def main() -> None:
  from holoradio.app import HoloRadio

  application = HoloRadio()
  scheduler = EventBasedScheduler(application, worker_thread_number=4)
  application.scheduler(scheduler)
  application.run()
