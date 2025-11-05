import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 
from src.features.pipe_config import preprocessing_pipeline
from src.utils.helpers import plot_confusion_matrix, plot_roc, classification_report_df
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay
from sklearn.pipeline import Pipeline
from io import BytesIO

def make_plots(X_train, y_train,cat_cols, num_cols) -> list:
    
    plot_list = list()

    preprocess = Pipeline(preprocessing_pipeline().steps[:-1])
    X_train_trans = preprocess.fit_transform(X_train, y_train)
    train= pd.concat([X_train, y_train], axis=1)
    train_processed = pd.concat([X_train_trans, y_train], axis=1)
    
    ### Cat_plot
    #! --------------------------------------------
    fig_cat = plt.figure(figsize=(18,8), dpi=150)
    fig_cat.subplots_adjust(wspace=0.4)
    for idx,col in enumerate(cat_cols, start=1): 
        plt.subplot(1,len(cat_cols),idx)
        sns.countplot(data=train, x=col, hue='Survived')
        plt.title('Survival vs ' + col.casefold().replace('_',' ') + 'variable', fontsize=14, fontweight='bold')
        plt.legend(labels = ['Did not survive','Did survive'])
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylabel('count',fontsize=14)
        plt.xlabel(col,fontsize=14)
        plt.tight_layout()
    #? --------------------------------------------
    plot_list.append(fig_cat)
    plt.close(fig_cat)
    #! --------------------------------------------
    # Missing Vals
    missing_pct = train.isnull().sum().sort_values(ascending=False) / len(train)

    fig_nans = plt.figure(figsize=(10,8), dpi=150)
    sns.barplot(y=missing_pct.index, x=missing_pct, color='red')
    plt.title('Train set missing values', fontsize=14, fontweight='bold')
    plt.ylabel('Variable',fontsize=14)
    plt.xlabel('% of missing',fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.tight_layout()
    #? -------------------------------------------- 
    plot_list.append(fig_nans)
    plt.close(fig_nans)
    #! --------------------------------------------
    #Age distributions
    missing_idx = train['Age'][train['Age'].isna()].index
    train_processed.loc[missing_idx, ]

    fig_ages, ax = plt.subplots(ncols=3,nrows=1,figsize=(22,8), dpi=200)
    sns.histplot(data=train, x='Age', label='pre-imputation',ax=ax[0])
    mean_age_before = train['Age'].mean()
    ax[0].axvline(mean_age_before, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age_before:.1f}')
    ax[0].set_title('Age distribution before imputation', fontsize=14)
    ax[0].legend()

    sns.histplot(data=train_processed, x='Age', label='post-imputation', color='green', ax=ax[1])
    mean_age_post_imp = train_processed['Age'].mean()
    ax[1].axvline(mean_age_post_imp, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age_post_imp:.1f}')
    ax[1].set_title('Age distribution after imputation', fontsize=14)
    ax[1].legend()

    sns.histplot(data=train_processed.loc[missing_idx, ], x='Age', label='imputed sample', bins=10, color='orange',ax=ax[2])
    mean_age_sample = train_processed.loc[missing_idx, ]['Age'].mean()
    ax[2].axvline(mean_age_sample, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age_sample:.1f}')
    ax[2].set_title('Distribution of imputed sample', fontsize=14)

    #? --------------------------------------------
    plot_list.append(fig_ages)
    plt.close(fig_ages)
    #! --------------------------------------------
    #DISTRIBUTIONS 1

    fig_num = plt.figure(figsize=(22,6))
    plt.subplots_adjust(wspace=0.4)
    for idx,col in enumerate(num_cols, start=1): 
        plt.subplot(1,len(num_cols),idx)
        sns.histplot(data=train_processed, x=col, hue='Survived', multiple = 'stack')
        plt.title('Distribution of ' + col.casefold().replace('_',' '))
        plt.legend(labels = ['Did not survive','Did survive'])
    #? --------------------------------------------
    plot_list.append(fig_num)
    plt.close(fig_num)
    #! --------------------------------------------

    return plot_list

def evaluate_plots(predictions, y_test):
    models = predictions.columns
    cm_fig = plt.figure(figsize=(18,10), dpi=150)
    for i,model in enumerate(models,start=1):
        # buf=BytesIO()
        ax = cm_fig.add_subplot(2,3,i)
        ConfusionMatrixDisplay.from_predictions(y_test,
                                                 predictions[model],
                                                 ax=ax)
        ax.set_title(f'Confusion matrix: {model}')
    plt.close(cm_fig)
    roc_fig = plt.figure(figsize=(18,10), dpi=150)
    for i,model in enumerate(models,start=1):
        # buf=BytesIO()
        ax = roc_fig.add_subplot(2,3,i)
        RocCurveDisplay.from_predictions(y_test,
                                                 predictions[model],
                                                 ax=ax)
        ax.set_title(f'Confusion matrix: {model}')
    plt.close(roc_fig)
    
    return cm_fig, roc_fig


def test_performance(predictions, y_test):
    fill = pd.DataFrame()
    full_reports = dict()
    for i in predictions.columns:
        full_reports[i] = classification_report_df(y_test, predictions.loc[:,i])
        fill[i] = classification_report_df(y_test, predictions.loc[:,i]).loc['weighted avg']
    fill = fill.drop('support', axis=0).T
    fig = plt.figure(figsize=(18,8),dpi=150)
    for (num,i), color in zip(enumerate(fill.columns, start=1),['red','steelblue','orange']):
        ax = fig.add_subplot(1,3,num)
        fill_sorted = fill.sort_values(by=i, ascending=False)
        sns.barplot(y=fill_sorted[i], x=fill_sorted.index, color=color, ax=ax)
        plt.ylim(0.75,0.87)
        plt.title('Weighted average '+i+' on test set', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('model',fontsize=14)
        plt.ylabel(i,fontsize=14)
    plt.close(fig)


    return fig, full_reports
