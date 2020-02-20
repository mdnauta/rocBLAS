#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import os
import re
import sys

sys.path.append('../../../clients/common/')
import rocblas_gentest as gt

import commandrunner as cr

# TODO: Should any of these ignored arguments be passed on?
IGNORE_YAML_KEYS = [
        'KL',
        'KU',
        'incd',
        'incb',
        'alphai',
        'betai',
        'norm_check',
        'unit_check',
        'timing',
        'algo',
        'solution_index',
        'flags',
        'workspace_size',
        'initialization',
        'category',
        'known_bug_platforms',
        'name',
        'c_noalias_d',
        'flops',
        'mem',
        'samples',
        'a_type',
        'b_type',
        'c_type',
        'd_type',
        'stride_x',
        'stride_y',
        'ldd',
        'stride_a',
        'stride_b',
        'stride_c',
        'stride_d',
        ]
REGULAR_YAML_KEYS = [
        'batch_count',
        'function',
        'compute_type',
        'incx',
        'incy',
        'alpha',
        'beta',
        'iters',
        #samples', TODO: Implement this functionality at a low level
        'transA',
        'transB',
        'side',
        'uplo',
        'diag',
        ]
SWEEP_YAML_KEYS = [
        'n',
        'm',
        'k',
        'lda',
        'ldb',
        'ldc',
        ]

# If an argument is not relevant to a function, then its value is set to '*'.
# We cannot pass a '*' to subsequent commands because it will, so that flag
# needs to be removed.
class StripStarsArgument(cr.ArgumentABC):
    def __init__(self, flag):
        cr.ArgumentABC.__init__(self)
        self.flag = flag

    def get_args(self):
        if self._value is None:
            return []
            #raise RuntimeError('No value set for {}'.format(self.flag))
        if self._value == '*': # If an asterisk is specified
            return [] # Just ignore the flag entirely
        return [self.flag, str(self._value)]

# TODO: handle this better
class IgnoreArgument(cr.ArgumentABC):
    def __init__(self, flag):
        cr.ArgumentABC.__init__(self)
        self.flag = flag

    def get_args(self):
        return []

class RocBlasArgumentSet(cr.ArgumentSetABC):
    def _define_consistent_arguments(self):
        self.consistent_args['n'             ] = StripStarsArgument('-n'             )
        self.consistent_args['m'             ] = StripStarsArgument('-m'             )
        self.consistent_args['k'             ] = StripStarsArgument('-k'             )
        self.consistent_args['batch_count'   ] = StripStarsArgument('--batch_count'  ) #
        self.consistent_args['function'      ] = StripStarsArgument('-f'             ) #
        self.consistent_args['compute_type'  ] = StripStarsArgument('-r'             ) # precision
        self.consistent_args['incx'          ] = StripStarsArgument('--incx'         )
        self.consistent_args['incy'          ] = StripStarsArgument('--incy'         )
        self.consistent_args['alpha'         ] = StripStarsArgument('--alpha'        )
        self.consistent_args['beta'          ] = StripStarsArgument('--beta'         )
        self.consistent_args['iters'         ] = StripStarsArgument('-i'             ) #
        self.consistent_args['lda'           ] = StripStarsArgument('--lda'          )
        self.consistent_args['ldb'           ] = StripStarsArgument('--ldb'          )
        self.consistent_args['ldc'           ] = StripStarsArgument('--ldc'          )
        self.consistent_args['transA'        ] = StripStarsArgument('--transposeA'   )
        self.consistent_args['transB'        ] = StripStarsArgument('--transposeB'   )
        #self.consistent_args['initialization'] = StripStarsArgument('-initialization') # Unused?
        self.consistent_args['side'          ] = StripStarsArgument('--side'         )
        self.consistent_args['uplo'          ] = StripStarsArgument('--uplo'         )
        self.consistent_args['diag'          ] = StripStarsArgument('--diag'         )
        self.consistent_args['device'        ] = cr.DefaultArgument('--device', 0    )

    def _define_variable_arguments(self):
        self.variable_args['output_file'] = cr.PipeToArgument()

    def __init__(self, **kwargs):
        cr.ArgumentSetABC.__init__(
                self, **kwargs
                )

    def get_full_command(self, run_configuration):
        exec_name = os.path.join(run_configuration.executable_directory, 'rocblas-bench')
        if not os.path.exists(exec_name):
            raise RuntimeError('Unable to find {}!'.format(exec_name))

        #self.set('nsample', run_configuration.num_runs)
        self.set('output_file', self.get_output_file(run_configuration))

        return [exec_name] + self.get_args()

    def collect_timing(self, run_configuration, data_type='time'):
        output_filename = self.get_output_file(run_configuration)
        rv = {}
        print('Processing {}'.format(output_filename))
        if os.path.exists(output_filename):
            lines = open(output_filename, 'r').readlines()
            us_vals = []
            gf_vals = []
            bw_vals = []
            gf_string = "rocblas-Gflops"
            bw_string = "rocblas-GB/s"
            us_string = "us"
            for i in range(0, len(lines)):
                if re.search(r"\b" + re.escape(us_string) + r"\b", lines[i]) is not None:
                    us_line = lines[i].strip().split(",")
                    index = [idx for idx, s in enumerate(us_line) if us_string in s][0] #us_line.index()
                    us_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
                if gf_string in lines[i]:
                    gf_line = lines[i].split(",")
                    index = gf_line.index(gf_string)
                    gf_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
                if bw_string in lines[i]:
                    bw_line = lines[i].split(",")
                    index = bw_line.index(bw_string)
                    bw_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
            if len(us_vals) > 0:
                rv['Time (microseconds)'] = us_vals
            if len(bw_vals) > 0:
                rv['Bandwidth (GB/s)'] = bw_vals
            if len(gf_vals) > 0:
                rv['Flops (GFlops/s)'] = gf_vals
        else:
            print('{} does not exist'.format(output_filename))
        return rv


class YamlData:

    def __init__(self, config_file):
        self.config_file = config_file
        self.test_cases = []
        self.execute_run()

    def reorder_data(self):
        old_data = self.test_cases
        new_data = []
        names = []
        for test in old_data:
            name = test['function']
            precision = test['compute_type']
            side = test['side']
            if (name,precision) not in names: # TODO: This will always be true because "side" is not in the tuple.
                type = [ x for x in old_data if x['function']==name and x['compute_type'] == precision and x['side'] == side ]
                new_data.append(type)
                names.append((name,precision, side))
        self.test_cases = new_data

    #Monkey Patch
    def write_test(self, test):
        self.test_cases.append(test)

    #Monkey Patch
    def process_doc(self, doc):
        """Process one document in the YAML file"""

        # Ignore empty documents
        if not doc or not doc.get('Tests'):
            return

        # Clear datatypes and params from previous documents
        gt.datatypes.clear()
        gt.param.clear()

        # Return dictionary of all known datatypes
        gt.datatypes.update(gt.get_datatypes(doc))

        # Arguments structure corresponding to C/C++ structure
        gt.param['Arguments'] = type('Arguments', (gt.ctypes.Structure,),
                                {'_fields_': gt.get_arguments(doc)})

        # Special names which get expanded as lists of arguments
        gt.param['dict_lists_to_expand'] = doc.get('Dictionary lists to expand') or ()

        # Lists which are not expanded
        gt.param['lists_to_not_expand'] = doc.get('Lists to not expand') or ()

        # Defaults
        defaults = doc.get('Defaults') or {}

        default_add_ons = {'m': 1, 'M': 1, 'n': 1, 'N': 1, 'k': 1, 'K': 1, 'lda': 1, 'ldb': 1, 'ldc': 1, 'LDA': 1, 'LDB': 1, 'LDC': 1, 'iters': 1, 'flops': '', 'mem': '', 'samples': 1, 'step_mult': 0}
        defaults.update(default_add_ons)

        # Known Bugs
        gt.param['known_bugs'] = doc.get('Known bugs') or []

        # Functions
        gt.param['Functions'] = doc.get('Functions') or {}

        # Instantiate all of the tests, starting with defaults
        for test in doc['Tests']:
            case = defaults.copy()
            case.update(test)
            gt.generate(case, gt.instantiate)

    def import_data(self):
        gt.args['includes'] = []
        gt.args['infile'] = self.config_file
        gt.write_test = self.write_test
        for doc in gt.get_yaml_docs():
            self.process_doc(doc)

    def execute_run(self):
        self.import_data()
        self.reorder_data()

<<<<<<< HEAD

=======
>>>>>>> aa66df8983153d91c1890c4b1b3fbbdabff9c16e
class RocBlasYamlComparison(cr.Comparison):
    def __init__(self, test_yaml, data_type, **kwargs):
        def get_function_prefix(compute_type):
            if '32_r' in compute_type:
                return 's'
            elif '64_r' in compute_type:
                return 'd'
            elif '32_c' in compute_type:
                return 'c'
            elif '64_c' in compute_type:
                return 'z'
            elif 'bf16_r' in compute_type:
                return 'bf'
            elif 'f16_r' in compute_type:
                return 'h'
            else:
                print('Error - Cannot detect precision preFix: ' + compute_type)
        cr.Comparison.__init__(self,
            description=get_function_prefix(test_yaml[0]['compute_type']) + test_yaml[0]['function'].split('_')[0] + data_type.capitalize() + ' Performance',
            **kwargs)

        for test in test_yaml:
            argument_set = RocBlasArgumentSet()
            all_inputs = {key:test[key] for key in test if not key in IGNORE_YAML_KEYS} # deep copy and cast to dict
            # regular keys have a direct mapping to the benchmark executable
            for key in REGULAR_YAML_KEYS:
                argument_set.set(key, all_inputs.pop(key))
            # step_size and step_mult are special, the determine how to sweep variables
            step_size = int(all_inputs.pop('step_size')) if 'step_size' in all_inputs else 10 #backwards compatiable default
            step_mult = (int(all_inputs.pop('step_mult')) == 1) if 'step_mult' in all_inputs else False
            if step_size == 1 and step_mult:
                raise ValueError('Cannot increment by multiplying by one.')
            sweep_lists = {}
            for key in SWEEP_YAML_KEYS:
                key_min = int(all_inputs.pop(key))
                key_max = int(all_inputs.pop(key.upper()))
                key_values = []
                while key_min <= key_max:
                    key_values.append(key_min)
                    key_min = key_min*step_size if step_mult else key_min+step_size
                sweep_lists[key] = key_values
            sweep_lengths = {key:len(sweep_lists[key]) for key in sweep_lists}
            max_sweep_length = max(sweep_lengths.values())
            for sweep_idx in range(max_sweep_length):
                sweep_argument_set = argument_set.get_deep_copy()
                for key in sweep_lists:
                    if sweep_lengths[key] == max_sweep_length:
                        sweep_argument_set.set(key, sweep_lists[key][sweep_idx])
                self.add(sweep_argument_set)
            if len(all_inputs) > 0:
                print('WARNING - The following values were unused: {}'.format(all_inputs))
        self.data_type = data_type

data_type_classes = {}
class TimeComparison(RocBlasYamlComparison):
    def __init__(self, **kwargs):
        RocBlasYamlComparison.__init__(self, data_type='time', **kwargs)
data_type_classes['time'] = TimeComparison

class FlopsComparison(RocBlasYamlComparison):
    def __init__(self, **kwargs):
        RocBlasYamlComparison.__init__(self, data_type='gflops', **kwargs)
#data_type_classes['gflops'] = FlopsComparison

class BandwidthComparison(RocBlasYamlComparison):
    def __init__(self, **kwargs):
        RocBlasYamlComparison.__init__(self, data_type='bandwidth', **kwargs)
#data_type_classes['bandwidth'] = BandwidthComparison

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', '--num-runs', default=10, type=int,
                        help='Number of times to run each test.')
    parser.add_argument('--data-types', default=data_type_classes.keys(), nargs='+',
                        choices = data_type_classes.keys(),
                        help='Types of data to generate plots for.')
    parser.add_argument('-I', '--input-yaml', required=True,
                        help='rocBLAS input yaml config.')
    user_args = cr.parse_input_arguments(parser)

    command_runner = cr.CommandRunner(user_args)

    command_runner.setup_system()

    #load yaml then create fig for every test
    with open(user_args.input_yaml, 'r') as f:
        data = YamlData(f)
        f.close()

    comparisons = []

    #setup tests sorted by their respective figures
    for test_yaml in data.test_cases:
        for data_type in user_args.data_types:
            data_type_cls = data_type_classes[data_type]
            comparison = data_type_cls(test_yaml = test_yaml)
            comparisons.append(comparison)

    command_runner.add_comparisons(comparisons)
    command_runner.main()
