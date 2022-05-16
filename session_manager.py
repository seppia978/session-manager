import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import json
import warnings
from abc import ABC, abstractmethod
import images_utils_978 as IMUT


class _Session_Manager(ABC):

    def __init__(self):
        ...
    
    @abstractmethod
    def update(self,**kwargs):
        raise NotImplementedError


class Session_Manager(_Session_Manager):

    def __init__(self,
                 name,
                 **kwargs
                 ):
        super().__init__()
        self.name,self.internals=name,dict()
        self.update(**kwargs)


    def update(self,**kwargs):

        self.internals.update(**kwargs)
        for k,v in kwargs.items():
            self.__setattr__(k,v)

    def __repr__(self):
        ret = ''
        for k,v in self.internals.items():
            ret += f'{k}: {v}\n'

        return ret

    def __getitem__(self, item):
        return self.internals[item]


    def percentage(self, idx, total_length):
        if idx == 0 and total_length == 1:
            return 100.
        else:
            percentage = round(min(idx*100/total_length, 100),2)
            return percentage

    def print_status(self, i, num_imgs):
        # return f'{self.current_status.current_images["print_ids"]}\t[{self.percentage(i, num_imgs)}%]\t{self.attribution_method.name}'
        return "{:<10} {:<10} {}".format(self.current_status.current_images["print_ids"],f'[{self.percentage(i, num_imgs)}%]',self.attribution_method.name)

    def evaluator_save_files(self, session):
        dataset = session.dataset
        imgs = session.current_status.current_images['names']
        sm = session.current_status.saliency_maps
        em = session.current_status.explanation_maps
        raws = session.current_status.raws
        path = session.outpath
        class_names = session.current_status.class_names

        for i, im in enumerate(imgs):
            base_path = os.path.join(path, im)

            if not os.path.isdir(base_path):
                os.mkdir(base_path)

            F.to_pil_image(sm[i].squeeze(0).cpu().detach()).save(os.path.join(base_path,"saliency_map.png"))

            with open(os.path.join(base_path,"class.txt"), 'w') as f:
                f.write(f'{class_names[i]} {dataset.name}\n')
            raws[i].save(os.path.join(base_path,"image.png"))
            F.to_pil_image(em[i].squeeze(0).cpu().detach()).save(os.path.join(base_path,"explanation_map.png"))

            img = IMUT.apply_transform(
                IMUT.load_image(os.path.join(base_path,"image.png")),
                normalize=False,
                size = session.current_status.input[i].shape[-1]
            ).squeeze().permute(1,2,0).cpu().detach().numpy()

            #cv2.imread(os.path.join(base_path,"image.png"))[:, :, ::-1]
            sal = sm[i].squeeze(0).cpu().detach().numpy()
            sizes = img.shape
            height = float(sizes[0])
            width = float(sizes[1])

            fig = plt.figure()
            fig.set_size_inches(width / height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            my_cmap1 = plt.cm.get_cmap('jet')
            ax.imshow(img, origin='upper')
            ax.imshow(sal, origin='upper', extent=[0, img.shape[1], img.shape[0], 0], alpha=0.5,
                      cmap=my_cmap1)
            plt.savefig(os.path.join(base_path,"explanation_map_1.png"), dpi=height)
            plt.close(fig)

            # fig = plt.figure()
            # fig.set_size_inches(width / height, 1, forward=False)
            # ax = plt.Axes(fig, [0., 0., 1., 1.])
            # ax.set_axis_off()
            # fig.add_axes(ax)
            #
            # my_cmap2 = plt.cm.get_cmap('jet')
            # ax.imshow(img, origin='upper')
            # ax.imshow(sal, origin='upper', extent=[0, img.shape[1], img.shape[0], 0], alpha=0.5,
            #           cmap=my_cmap2)
            # plt.savefig(os.path.join(base_path,"explanation_map_2.png"), dpi=height)
            # plt.close(fig)

    def save_single_image_results_json(self, filename, session):
        all_results = session.current_status.all_results
        path = session.outpath

        if len(list(all_results.values())[0]) \
                != len(session.current_status.current_images['names']):
            warnings.warn(f'all_res len: {len(list(all_results.values())[0])}\
            , images len: {len(session.current_status.current_images["names"])}')

        for idx, name in enumerate(session.current_status.current_images['names']):
            base_path = os.path.join(path, name)
            try:
                pass
                # with open(os.path.join(base_path, f"{filename}.json"), 'r') as fp:
                #     all_results = {**all_results, **json.load(fp)}
            except:
                pass


            with open(os.path.join(base_path, f"{filename}.json"), 'w') as fp:
                to_dump = {k:[v[idx]] for k,v in all_results.items()}
                json.dump(to_dump, fp, indent=4)

    def save_all_images_results_json(self, filename, session):
        m = session.current_status.current_metric
        path = session.outpath

        with open(os.path.join(path, f"{filename}.json"), 'w') as fp:
            json.dump(m.get_result(), fp, indent=4)
