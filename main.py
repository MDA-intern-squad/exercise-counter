import sys
import mpCustom as ct

dirName = sys.argv[1]
actionName = sys.argv[2]

bootstrap_helper = ct.BootstrapHelper(
    images_in_folder=f'./data/{dirName}/in',
    images_out_folder=f'./data/{dirName}/out',
    csvs_out_folder=f'./data/{dirName}/dist',
    pose_class_name=actionName
)

bootstrap_helper.bootstrap(per_pose_class_limit=None)
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
