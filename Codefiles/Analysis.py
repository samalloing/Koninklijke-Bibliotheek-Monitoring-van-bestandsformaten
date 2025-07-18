from peak_estimation import fit_bass_minimax, polynomial, spline, fit_sine_minimax, frame_bound, reorient_frame, bootstrap_frame_bound
from open_kb_files import add_zeroes  
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from os import path, makedirs
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

window_size = 8
na_metaframe = pd.read_csv(r'Codefiles\Repositories deduplicated\na_metaframe.csv', index_col='Date')
kb_metaframe = pd.read_csv(r'Codefiles\Repositories deduplicated\kb_metaframe.csv', index_col='Date')
dans_metaframe = pd.read_csv(r'Codefiles\Repositories deduplicated\dans_metaframe.csv', index_col='productiondate')

all_metaframe = pd.read_csv(r'Codefiles\Repositories deduplicated\all_metaframe.csv', index_col='Date') 
ext_sources = pd.concat([kb_metaframe[['Extension']].assign(source='kb'),
                         dans_metaframe[['Extension']].assign(source='dans'),
                         pd.DataFrame({'Extension' : na_metaframe[['ExternalSignature']].values.flatten()}).assign(source='na')
                        ])

ext_counts = ext_sources.drop_duplicates().groupby('Extension')['source'].nunique()
# Keep only Extensions that appear in more than one source
valid_extensions = ext_counts[ext_counts > 1].index
# Filter all_metaframe to keep only valid Extensions
all_metaframe = all_metaframe[all_metaframe['Extension'].isin(valid_extensions) | all_metaframe['Extension'].isin(['bz2', 'jar', 'java'])]

del ext_sources
del ext_counts
del valid_extensions

for (df, metaname) in ((na_metaframe, 'na_metaframe'), (kb_metaframe, 'kb_metaframe'), (dans_metaframe, 'dans_metaframe'), (all_metaframe, 'all_metaframe')): 
    df.index.freq = 'MS'
    df.index = pd.to_datetime(df.index)
del df

def bootstrap_validation(repo, label, f_type, validation_frame, window_size = window_size, bootstrap_samples=1000):
    bootstrap_frame = pd.DataFrame(columns=['Repository', 'Extension', 'Method', 'Bootstrap Trial', 'Maximum index', 'Estimated peak index', 'Peak Error', 'Max Value', 'Left inflection', 'Right inflection'])
    condit = (validation_frame['Repository']==label) & (validation_frame['Extension']==f_type)
    for trial in range(1, bootstrap_samples + 1):
        unfiltered_repo, unfiltered_scaler, unfiltered_left_bound = bootstrap_frame_bound(repo.copy())
        max_value = unfiltered_repo['Count'].max()
        max_index = max(unfiltered_repo.loc[unfiltered_repo['Count']==max_value].index.values) if max_value else None
        max_value = unfiltered_scaler.inverse_transform(np.array([max_value]).reshape(-1, 1))[0][0]
        for (method, fit_results) in [('Bass', fit_bass_minimax(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound, tuple(validation_frame.loc[condit & (validation_frame['Method'] == 'Bass'), ['Parameter 1', 'Parameter 2', 'Parameter 3']].values.flatten()))),
                                      ('Sinus', fit_sine_minimax(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound, tuple(validation_frame.loc[condit & (validation_frame['Method'] == 'Sinus'), ['Parameter 1', 'Parameter 2', 'Parameter 3']].values.flatten()))),
                                      ('Polynoom', polynomial(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound, tuple(validation_frame.loc[condit & (validation_frame['Method']=='Polynoom'), ['Parameter 1', 'Parameter 2']].values.flatten()))),
                                      ('Spline', spline(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound, tuple(validation_frame.loc[condit & (validation_frame['Method']=='Spline'), ['Parameter 1', 'Parameter 2', 'Parameter 3']].values.flatten())))
                                      ]:
            frame, peak, inflection_points, _ = fit_results
            if max_index is not None:
                delta = peak - max_index
                to_add1 = pd.DataFrame({'Repository': [label], 'Extension': [f_type], 'Method': [method], 'Bootstrap Trial': [trial], 'Maximum index': [max_index],
                                        'Estimated peak index': [peak], 'Peak Error': [delta], 'Max Value' : [max_value], 'Left inflection': [inflection_points[0]], 'Right inflection': [inflection_points[1]]})
                bootstrap_frame = pd.concat([bootstrap_frame, to_add1])
            else:
                to_add2 = pd.DataFrame({'Repository': [label], 'Extension': [f_type], 'Method': [method], 'Bootstrap Trial': [trial], 'Maximum index': [np.nan],
                                        'Estimated peak index': [peak], 'Peak Error': [np.nan], 'Max Value' : [np.nan], 'Left inflection': [inflection_points[0]], 'Right inflection': [inflection_points[1]]})
                bootstrap_frame = pd.concat([bootstrap_frame, to_add2])
            del frame
    return bootstrap_frame

def plot(repo, label, f_type, save_dir, all_dir, window_size = window_size):
    unfiltered_repo, unfiltered_scaler, unfiltered_left_bound = frame_bound(repo.copy())
    date_index = unfiltered_repo['Date']
    delta = (pd.to_datetime(date_index.max(), unit='D') - pd.to_datetime(date_index.min(), unit='D')) / pd.Timedelta(days=1)
    if delta < (8 * 365 + 2 * 366): # tien jaar in dagen.
        return pd.DataFrame()
    fig, ax = plt.subplots(figsize=(150, 30), layout='constrained')
    ax.set_title(f"{label} – {f_type}", fontdict={'fontsize' : 200})
    ax.plot(repo.index, repo.values, label=label, linewidth=12)
    ax.set_ylabel('Aangemaakte bestanden per maand', fontsize=100, labelpad=50)

    locator = mdates.YearLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=80)
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, which='minor', axis='x', linestyle='--', color='darkgray', linewidth = "3")

    ax.tick_params(axis='x', which='major', labelsize=100)
    ax.tick_params(axis='y', which='major', labelsize=100)

    max_value = repo['Count'].max()
    if len(repo.loc[repo['Count'] == max_value].index.values) > 1:
        print('Maximum value not unique!')
        
    bounded_repo_max = repo.loc[repo['Count'] == max_value].index.values[0]
    repo, scaler, left_bound = frame_bound(repo)
    validation_frame = pd.DataFrame(columns=['Repository', 'Extension', 'Method', 'Maximum index', 'Estimated peak index', 'Left inflection', 'Right inflection', 'Peak Error'])
    bounded_repo_date = unfiltered_repo[['Date']]
    colors = ('green', 
              'yellow', 
              'red', 
              'orchid')
    for idx, (method, fit_results) in enumerate([('Bass', fit_bass_minimax(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound)),
                                                 ('Sinus', fit_sine_minimax(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound)),
                                                 ('Polynoom', polynomial(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound)),
                                                 ('Spline', spline(unfiltered_repo.index.values, unfiltered_repo['Count'].values, unfiltered_scaler, unfiltered_left_bound))                                                  
                                                ]):
        frame, peak, inflection_points, (param1, param2, param3) = fit_results
        if method != 'Spline':
            peak = peak + unfiltered_left_bound
        reorient, inflection_points, max_index, _ = reorient_frame(bounded_repo_date, frame, peak, *inflection_points)
        color = colors[idx % len(colors)]

        if max_index is not None:
            to_add3 = pd.DataFrame({'Repository': [label], 'Extension': [f_type], 'Method': [method], 'Maximum index': [max_index],
                                    'Estimated peak index': [peak], 'Left inflection': [inflection_points[0]], 'Right inflection': [inflection_points[1]],
                                    'Peak Error': [(pd.to_datetime(max_index).year - pd.to_datetime(bounded_repo_max).year) * 12 + (pd.to_datetime(max_index).month - pd.to_datetime(bounded_repo_max).month)],
                                    'Parameter 1': [param1], 'Parameter 2': [param2], 'Parameter 3': [param3]})
            validation_frame = pd.concat([validation_frame, to_add3])
            ax.scatter(inflection_points[0], max_value, color=color, s=600, label='_nolegend_')
            ax.scatter(max_index, max_value, color=color, s=800, label='_nolegend_')
            ax.scatter(inflection_points[1], max_value, color=color, s=600, label='_nolegend_')
        else:
            to_add4 = pd.DataFrame({'Repository': [label], 'Extension': [f_type], 'Method': [method], 'Maximum index': [np.nan],
                                    'Estimated peak index': [peak], 'Left inflection': [inflection_points[0]], 'Right inflection': [inflection_points[1]],
                                    'Peak Error': [np.nan], 'Parameter 1': [param1], 'Parameter 2': [param2], 'Parameter 3': [param3]})
            validation_frame = pd.concat([validation_frame, to_add4])
        ax.plot(reorient.index, reorient['Count'], color=color, label=method, linewidth=12)
        del frame
    ax.legend(('Data', 
               'Bass', 
               'Sinus', 
               'Polynoom', 
               'Spline'), fontsize=100)

    fig.tight_layout()
    makedirs(save_dir, exist_ok=True)
    save_entry = path.join(save_dir, f"{f_type} - {label} monitoring.png")
    fig.savefig(save_entry)
    makedirs(all_dir, exist_ok=True)
    save_entry = path.join(all_dir, f"{f_type} - {label} monitoring.png")
    fig.savefig(save_entry)
    save_entry = f"Codefiles\Validations\Validation plots\\{label}\\{f_type} - {label}"
    fig.savefig(r''.format(save_entry))
    plt.show()
    return validation_frame
validation_frame = pd.DataFrame(columns=['Repository', 'Extension', 'Method', 'Peak Error'])
bootstrap_frame = pd.DataFrame(columns=['Repository', 'Extension', 'Method', 'Bootstrap Trial', 'Maximum index', 'Estimated peak index', 'Peak Error', 'Max Value', 'Left inflection', 'Right inflection'])

for label, repo, col in [('KB', kb_metaframe, 'Extension'), 
                         ('DANS', dans_metaframe, 'Extension'), 
                         ('NA', na_metaframe, 'ExternalSignature'),
                         ('All', all_metaframe, 'Extension')]:
    save_dir = f"Codefiles\\Validations\\Validation plots\\{label}"
    all_dir = f"Codefiles\\Validations\\Ungrouped Validation plots"
    order = repo.groupby([col])[col].count().reset_index(name='Count').sort_values(['Count'], ascending=False)[col].values
    print(order)
    for f_type in order: # repo[col].unique()
        repo_subframe = add_zeroes(repo.loc[(repo[col] == f_type), 'Count'])
        f_type_row = plot(repo_subframe, label, f_type, save_dir, all_dir)
        if f_type_row.shape == (0, 0):
            continue
        else:
            validation_frame = pd.concat([validation_frame, f_type_row])
            bootstrap_frame = pd.concat([bootstrap_frame, bootstrap_validation(repo_subframe, label, f_type, validation_frame)])
del repo_subframe
validation_frame.to_csv(r"Codefiles\\Validations\\validation_frame.csv")
bootstrap_frame.to_csv(r"Codefiles\\Validations\\bootstrap_frame.csv")