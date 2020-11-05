import json
import os
import string

os.chdir(r'D:\projects\FewRel')


def format_sample(sample):
    """
    将单个样本转化为方便阅读的字符串
    
    Parameters
    ----------
    sample: dict{'tokens': list[str], 'h': [str, str, list[list[int]]], 't': [str, str, list[list[int]]]}
        FewRel数据集中的单个句子

    Returns
    ----------
    res: sentence
        方便阅读的字符串
        
    """
    for idx in sample['h'][2]:
        sample['tokens'][idx[0]] = '{' + sample['tokens'][idx[0]]
        sample['tokens'][idx[-1]] = sample['tokens'][idx[-1]] + '}'
        
    for idx in sample['t'][2]:
        sample['tokens'][idx[0]] = '[' + sample['tokens'][idx[0]]
        sample['tokens'][idx[-1]] = sample['tokens'][idx[-1]] + ']'
     
    sentence = ''
    sz = len(sample['tokens'])
    
    for i in range(sz):
        cur = sample['tokens'][i]
        nxt = None
        if i + 1 < sz:
            nxt = sample['tokens'][i + 1]
        sentence += cur
        if nxt is not None and nxt not in string.punctuation and cur not in '([-':
            sentence += ' '
    
    return sentence

def write_class(data, class_id, filename, pid2name):
    """
    将FewRel中一个类别的数据以便于阅读的形式写入class_id.view文件
    
    Parameters
    ----------
    data: dict
        数据集
    class_id: str
        类别ID
    filename: str
        写入文件的路径
    pid2name: dict{str: [str, str]}
        类别id: [类别名, 类别描述]
    
    Returns
    ----------
    None
        
    """
    with open(filename, mode='w', encoding="utf-8") as fp:
        class_name, description = pid2name[class_id]
        print('Relation name:', class_name, file=fp)
        print('Description:', description, file=fp)
        print(file=fp)
        for i in range(len(data[class_id])):
            print(format_sample(data[class_id][i]), file=fp)
            print(file=fp)


if __name__ == '__main__':
    train_path = os.path.join('data', 'val_wiki.json')
    train_data = json.load(open(train_path))
    
    pid2name = json.load(open(r'D:\projects\FewRel\data\pid2name.json'))
    class_id = 'P463'
    view_filename = os.path.join('data', class_id + '.view')
    write_class(train_data, class_id, view_filename, pid2name)
    