from zipfile import ZipFile

path = 'data'
file_name = 'by_field.zip'

if __name__ == '__main__':

    with ZipFile(f'{path}\\{file_name}', 'r') as zip:
        zip.printdir()