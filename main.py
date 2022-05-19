from RPA.Browser.Selenium import Selenium

br = Selenium()


def open_quandri():
    br.open_available_browser("https://www.quandri.io")


if __name__ == "__main__":
    open_quandri()
