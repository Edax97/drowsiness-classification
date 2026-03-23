from serial_send import SerialSender, ScannerListener
from dsm_status import DSMstatus
import time

BAUDRATE = 115200
MINIMUM_SEND_INTERVAL=0.5
SEND_INTERVAL=5

if __name__ == "__main__":    
    serial_sender = SerialSender("/tmp/ttyV0", BAUDRATE)
    scanner_listener = ScannerListener()

    # Test 1: Sent DSM status every 5 seconds PASS
    #dsm_status = DSMstatus(SEND_INTERVAL, MINIMUM_SEND_INTERVAL)
    #dsm_status.set_status(driver=True)

    # Test 2: Never send DSM status, just scannings PASS
    #dsm_status = DSMstatus(SEND_INTERVAL, MINIMUM_SEND_INTERVAL)
    #dsm_status.set_status(driver=False)

    # Test 3: Sent DSM only when change in status (or every hour) PASS
    #TEST=3
    #dsm_status = DSMstatus(3600, 1)
    #dsm_status.set_status(driver=False)

    # Test 4: DSM sent every 30 seconds, or when change in status PASS
    TEST=3
    dsm_status = DSMstatus(30, 1)
    dsm_status.set_status(driver=True)

    while True:
        time.sleep(0.50)
        scanner_data = scanner_listener.get_data()
        if len(scanner_data) > 0:
            serial_sender.send_serial(scanner_data)
            if TEST == 3:
                match scanner_data:
                    case "TEL":
                        dsm_status.set_status(driver=True, telefono=True)
                    case "DEF":
                        dsm_status.set_status(driver=True)
                    case "CINTURON":
                        dsm_status.set_status(driver=True, cinturon=False)

        dsm_data = dsm_status.get_status_string()
        if len(dsm_data) > 0:
            serial_sender.send_serial(dsm_data)

        

    
