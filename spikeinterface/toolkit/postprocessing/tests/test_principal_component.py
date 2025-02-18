import shutil
from pathlib import Path

import numpy as np

from spikeinterface import extract_waveforms, WaveformExtractor
from spikeinterface.extractors import toy_example

from spikeinterface.toolkit.postprocessing import WaveformPrincipalComponent, compute_principal_components


def setup_module():
    for folder in ('toy_rec_1seg', 'toy_sorting_1seg', 'toy_waveforms_1seg',
                   'toy_rec_2seg', 'toy_sorting_2seg', 'toy_waveforms_2seg'):
        if Path(folder).is_dir():
            shutil.rmtree(folder)

    recording, sorting = toy_example(num_segments=2, num_units=10)
    recording = recording.save(folder='toy_rec_2seg')
    sorting = sorting.save(folder='toy_sorting_2seg')
    we = extract_waveforms(recording, sorting, 'toy_waveforms_2seg',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)

    recording, sorting = toy_example(num_segments=1, num_units=10, num_channels=12)
    recording = recording.save(folder='toy_rec_1seg')
    sorting = sorting.save(folder='toy_sorting_1seg')
    we = extract_waveforms(recording, sorting, 'toy_waveforms_1seg',
                           ms_before=3., ms_after=4., max_spikes_per_unit=500,
                           n_jobs=1, chunk_size=30000)


def test_WaveformPrincipalComponent():
    we = WaveformExtractor.load_from_folder('toy_waveforms_2seg')
    unit_ids = we.sorting.unit_ids
    num_channels = we.recording.get_num_channels()
    pc = WaveformPrincipalComponent(we)

    for mode in ('by_channel_local', 'by_channel_global'):
        pc.set_params(n_components=5, mode=mode)
        print(pc)
        pc.run()
        for i, unit_id in enumerate(unit_ids):
            proj = pc.get_projections(unit_id)
            # print(comp.shape)
            assert proj.shape[1:] == (5, 4)

        # import matplotlib.pyplot as plt
        # cmap = plt.get_cmap('jet', len(unit_ids))
        # fig, axs = plt.subplots(ncols=num_channels)
        # for i, unit_id in enumerate(unit_ids):
        # comp = pca.get_components(unit_id)
        # print(comp.shape)
        # for chan_ind in range(num_channels):
        # ax = axs[chan_ind]
        # ax.scatter(comp[:, 0, chan_ind], comp[:, 1, chan_ind], color=cmap(i))
        # plt.show()

    for mode in ('concatenated',):
        pc.set_params(n_components=5, mode=mode)
        print(pc)
        pc.run()
        for i, unit_id in enumerate(unit_ids):
            proj = pc.get_projections(unit_id)
            assert proj.shape[1] == 5
            # print(comp.shape)

    all_labels, all_components = pc.get_all_components()

    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap('jet', len(unit_ids))
    # fig, ax = plt.subplots()
    # for i, unit_id in enumerate(unit_ids):
    # comp = pca.get_components(unit_id)
    # print(comp.shape)
    # ax.scatter(comp[:, 0], comp[:, 1], color=cmap(i))
    # plt.show()


def test_compute_principal_components_for_all_spikes():
    we = WaveformExtractor.load_from_folder('toy_waveforms_1seg')
    pc = compute_principal_components(we, load_if_exists=True)
    print(pc)

    pc_file1 = 'all_pc1.npy'
    pc.run_for_all_spikes(pc_file1, max_channels_per_template=7, chunk_size=10000, n_jobs=1)
    all_pc1 = np.load(pc_file1)

    pc_file2 = 'all_pc2.npy'
    pc.run_for_all_spikes(pc_file2, max_channels_per_template=7, chunk_size=10000, n_jobs=2)
    all_pc2 = np.load(pc_file2)

    assert np.array_equal(all_pc1, all_pc2)


def test_pca_models_and_project_new():
    from sklearn.decomposition import IncrementalPCA
    if Path('toy_waveforms_1seg/PCA').is_dir():
        shutil.rmtree('toy_waveforms_1seg/PCA')
        Path('toy_waveforms_1seg/params_pca.json').unlink()
    we = WaveformExtractor.load_from_folder('toy_waveforms_1seg')

    wfs0 = we.get_waveforms(unit_id=we.sorting.unit_ids[0])
    n_samples = wfs0.shape[1]
    n_channels = wfs0.shape[2]
    n_components = 5

    # local
    pc_local = compute_principal_components(we, n_components=n_components,
                                            load_if_exists=True, mode="by_channel_local")

    all_pca = pc_local.get_pca_model()
    assert len(all_pca) == we.recording.get_num_channels()

    # project
    new_waveforms = np.random.randn(100, n_samples, n_channels)
    new_proj = pc_local.project_new(new_waveforms)

    assert new_proj.shape == (100, n_components, n_channels)

    # global
    if Path('toy_waveforms_1seg/PCA').is_dir():
        shutil.rmtree('toy_waveforms_1seg/PCA')
        Path('toy_waveforms_1seg/params_pca.json').unlink()
    pc_global = compute_principal_components(we, n_components=n_components,
                                             load_if_exists=True, mode="by_channel_global")

    all_pca = pc_global.get_pca_model()
    assert isinstance(all_pca, IncrementalPCA)

    # project
    new_waveforms = np.random.randn(100, n_samples, n_channels)
    new_proj = pc_global.project_new(new_waveforms)

    assert new_proj.shape == (100, n_components, n_channels)

    # concatenated
    if Path('toy_waveforms_1seg/PCA').is_dir():
        shutil.rmtree('toy_waveforms_1seg/PCA')
        Path('toy_waveforms_1seg/params_pca.json').unlink()
    pc_concatenated = compute_principal_components(we, n_components=n_components,
                                                   load_if_exists=True, mode="concatenated")

    all_pca = pc_concatenated.get_pca_model()
    assert isinstance(all_pca, IncrementalPCA)

    # project
    new_waveforms = np.random.randn(100, n_samples, n_channels)
    new_proj = pc_concatenated.project_new(new_waveforms)

    assert new_proj.shape == (100, n_components)


if __name__ == '__main__':
    setup_module()
    # test_compute_principal_components_for_all_spikes()
    test_pca_models_and_project_new()
