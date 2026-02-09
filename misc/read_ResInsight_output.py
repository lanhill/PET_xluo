import numpy as np
import ast  # used to convert str to numerical

class read_ResInsight_output():
    def __init__(self, filename, keywords=None, dim=None, dtype=None):
        """
        Read data exported by ResInsight, often in the name of *.GRDECL
        :param filename: file name
        :param filename: key word
        :param dim: = (nx, ny, nz)
        :param dtype: data type
        """
        self.filename = filename
        if dim is None:
            self._get_dim()
        else:
            self.dim = dim
        self.keywords = keywords
        self.dtype = dtype

    def _get_dim(self, comments='--', delimiter=' ', end_sign='/'):
        dim = []
        with open(self.filename, 'r') as f:
            contents = f.readlines()
            start_idx = len(contents)
            for idx, txt in enumerate(contents):
                #line = txt if txt[-1] != '\n' else txt[:-1]
                line = txt.strip(' \t\n\r')
                if line.strip() == 'SPECGRID' or line.strip() == 'DIMENS':
                    start_idx = idx

                if line.strip() == end_sign:
                    break

                # don't read data before the key word
                if (line.strip()[0:len(comments)] != comments) and (idx > start_idx):
                    current_str = [el.strip(' \t\n\r') for el in line.split(delimiter) if el.strip(' \t\n\r') != '']
                    # dimension info contained in the first 3 positions
                    dim = [ast.literal_eval(el) for el in current_str[:3]]
                    break

        self.dim = dim

    def read(self, keywords=None, comments='--', delimiter=' ', end_sign='/'):
        if keywords is None:
            keywords = self.keywords

        readout = {}
        if hasattr(self, 'dim') and len(self.dim) > 0:
            readout['DIMENS'] = self.dim

        with open(self.filename, 'r') as f:
            contents = f.readlines()
            for keyword in keywords:
                found = False
                data = []
                start_idx = len(contents)
                for idx, txt in enumerate(contents):
                    #line = txt if txt[-1] != '\n' else txt[:-1]
                    line = txt.strip(' \t\n\r')
                    if line.strip() == keyword:
                        start_idx = idx
                        found = True

                    # don't read data before the key word
                    if found and (line.strip()[0:len(comments)] != comments) and (idx > start_idx):
                        current_str = [el.strip(' \t\n\r') for el in line.split(delimiter) if el.strip(' \t\n\r') != '']

                        # "len(current_str) > 0" to exclude empty lines
                        if len(current_str) > 0 and current_str[-1] != end_sign:
                            current_data = [ast.literal_eval(el) for el in current_str]
                        else:
                            if len(current_str) > 1:  # if current_str does not only contain "/" (or other end sign)
                                current_data = [ast.literal_eval(el) for el in current_str[:-1]]
                                #current_data = []
                                #for el in current_str:
                                #    if "." in el:
                                #        if el.replace(".", "").isnumeric():
                                #            current_data.extend([ast.literal_eval(el)])
                                #    else:
                                #        if el.isnumeric():
                                #            current_data.extend([int(ast.literal_eval(el))])

                            else:
                                current_data = []
                        data.extend(current_data)

                        # if a line ends with "/" (or other end sign), then stop after reading the data from this line
                        if len(current_str) > 0 and current_str[-1] == end_sign:
                            break
                if self.dtype is not None:
                    readout[keyword] = np.array(data).astype(self.dtype)
                else:
                    readout[keyword] = np.array(data)
        return readout

    # def read(self, comments='--', delimiter=' ', end_sign='/'):
    #     data = []
    #     with open(self.filename, 'r') as f:
    #         contents = f.readlines()
    #         start_idx = len(contents)
    #         for idx, txt in enumerate(contents):#@
    #             #line = txt if txt[-1] != '\n' else txt[:-1]
    #             line = txt.strip(' \t\n\r')
    #             if line.strip() == self.keyword:
    #                 start_idx = idx
    #
    #             if line.strip() == end_sign:
    #                 break
    #
    #             # don't read data before the key word
    #             if (line.strip()[0:len(comments)] != comments) and (idx > start_idx):
    #                 current_str = [el.strip(' \t\n\r') for el in line.split(delimiter) if el.strip(' \t\n\r') != '']
    #                 current_data = [ast.literal_eval(el) for el in current_str]
    #                 data.extend(current_data)
    #
    #         data = np.array(data).astype(self.dtype)
    #     return data.reshape(self.dim, order='F')

if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt

    test = False
    if test:
        fn = '/6TB/xluo/code/Python/PET_RamonCo/Examples/EnSURE/drogon/include/grid/drogon.grid'
        reader = read_ResInsight_output(fn, keywords=None, dim=None)
        dim = reader.dim
        print(dim)

        fn = '/6TB/xluo/code/Python/PET_RamonCo/Examples/EnSURE/drogon/include/grid/drogon.perm'
        reader = read_ResInsight_output(fn, keywords=None, dim=dim)
        data = np.log(np.reshape(reader.read(keywords=['PERMY'])['PERMY'], dim, order='F'))
        print(data.shape)

        plt.imshow(np.squeeze(data[:, :, 0]))
        plt.colorbar()
        plt.show()


        plt.imshow(np.squeeze(data[:, :, 10]))
        plt.colorbar()
        plt.show()

        plt.imshow(np.squeeze(data[:, :, 20]))
        plt.colorbar()
        plt.show()


    test = False
    if test:
        fn = '/6TB/xluo/code/Python/PET_RamonCo/Examples/3dBox/avo/grid/Grid.grdecl'
        reader = read_ResInsight_output(fn, keywords=None, dim=None)
        dim = reader.dim
        data = reader.read(keywords=['ZCORN', 'COORD'])
        print(data['ZCORN'].shape, data['COORD'].shape)

        fn = '/6TB/xluo/code/Python/PET_RamonCo/Examples/3dBox/avo/grid/DX.GRDECL'
        reader = read_ResInsight_output(fn, keywords=None, dim=dim)
        print(reader.dim)
        data = reader.read(keywords=['DX'])
        print(data['DX'].shape)

    # --
    test = True
    if test:
        os.chdir('/6TB/xluo/code/Python/PET_RamonCo/Examples/Smeaheia/input_data')
        fn = 'DZ.GRDECL'
        dims = (106, 174, 100)
        reader = read_ResInsight_output(fn, keywords=['DZ'], dim=dims)

        data = reader.read()['DZ']
        data = np.reshape(data, dims, order='F')
        print(data.shape)

        plt.imshow(np.squeeze(data[:, :, 0]))
        plt.show()

        plt.imshow(np.squeeze(data[:, :, 10]))
        plt.show()

        plt.imshow(np.squeeze(data[:, :, 20]))
        plt.show()