init_graph = function(graph, flag=true){
    graph = JSON.parse(graph);

    //给节点的属性进行操作
    graph.nodes.forEach(function (node) {
        node.itemStyle = null;
        node.symbolSize = 15;
        // node.value = node.symbolSize;
        if(flag == true)
            node.category = node.attributes.modularity_class;
        node.draggable = true;
    });
    return graph;
};

init_option = function(graph){
    var option;
    var categories = [];
    categories[0] = {
            name: '未激活节点',
            itemStyle: {
                color: '#2f4554',
                opacity: 0.9, //不透明度
            }
        }
        categories[1] = {
            name: '激活节点',
            itemStyle: {
                color: '#009688',
                opacity: 1, //不透明度
            },
            label: {
                fontSize: 20,
            },

        };
    //设置要生成的图的相关属性
    option = {
        title: {
            text: '',
            subtext: '',
            top: 'top',
            left: 'left'
        },
        tooltip: {},
        legend: [{
            // selectedMode: 'single',
            data: categories.map(function (a) {
                return a.name;
            })
        }],
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [
            {
                name: 'Les Miserables',
                type: 'graph',
                layout: 'none',
                //下面三项分别设置节点数据，边数据，种类数据
                data: graph.nodes,
                links: graph.links,
                categories: categories,
                roam: true,
                animation: false,
                focusNodeAdjacency: true,
                itemStyle: {
                    normal: {
                        borderColor: '#fff',
                        borderWidth: 1,
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.3)'
                    }
                },
                label: {
                    position: 'right',
                    formatter: '{b}'
                },
                lineStyle: {
                    color: 'rgba(0, 0, 0, 0.7)',
                    curveness: 0.3
                },
                emphasis: {
                    lineStyle: {
                        width: 10
                    }
                },
                force: {
                    repulsion: 270
                }
            }
        ]
    };
    return option;
}
