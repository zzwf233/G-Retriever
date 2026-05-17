from src.dataset.expla_graphs import ExplaGraphsDataset
from src.dataset.scene_graphs import SceneGraphsDataset
from src.dataset.scene_graphs_baseline import SceneGraphsBaselineDataset
from src.dataset.webqsp import WebQSPDataset
from src.dataset.webqsp_baseline import WebQSPBaselineDataset
from src.dataset.webqsp_ours import WebQSPOursDataset
from src.dataset.webqsp_ours_papercfg import WebQSPOursPaperCfgDataset
from src.dataset.webqsp_rand import WebQSPRandDataset


load_dataset = {
    'expla_graphs': ExplaGraphsDataset,
    'scene_graphs': SceneGraphsDataset,
    'scene_graphs_baseline': SceneGraphsBaselineDataset,
    'webqsp': WebQSPDataset,
    'webqsp_baseline': WebQSPBaselineDataset,
    'webqsp_ours': WebQSPOursDataset,
    'webqsp_ours_papercfg': WebQSPOursPaperCfgDataset,
    'webqsp_rand': WebQSPRandDataset,
}
