#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import sys
import os

sys.path.append('../../../clients/common/')
import rocblas_gentest as gt

import commandrunner as cr

# These are required to translate YAML into arguments to timing.py
def do_nothing(input_str):
    return input_str
def equal_to_one(input_str):
    return input_str == 1

USED_YAML_KEYS_FMT = {
        'n':              do_nothing,
        'N':              do_nothing,
        'm':              do_nothing,
        'M':              do_nothing,
        'k':              do_nothing,
        'K':              do_nothing,
        'batch_count':    do_nothing,
        'function':       do_nothing,
        'compute_type':   do_nothing,
        'incx':           do_nothing,
        'incy':           do_nothing,
        'alpha':          do_nothing,
        'beta':           do_nothing,
        'iters':          do_nothing,
        'samples':        do_nothing,
        'lda':            do_nothing,
        'ldb':            do_nothing,
        'ldc':            do_nothing,
        'LDA':            do_nothing,
        'LDB':            do_nothing,
        'LDC':            do_nothing,
        'transA':         do_nothing,
        'transB':         do_nothing,
        #'initialization': do_nothing, # Unused?
        'step_size':      do_nothing,
        'step_mult':      equal_to_one, # There always has to be one exception...
        'side':           do_nothing,
        'uplo':           do_nothing,
        'diag':           do_nothing,
}

# If an argument is not relevant to a function, then its value is set to '*'.
# We cannot pass a '*' to subsequent commands because it will, so that flag
# needs to be removed.
class StripStarsArgument(cr.ArgumentABC):
    def __init__(self, flag):
        cr.ArgumentABC.__init__(self)
        self.flag = flag

    def get_args(self):
        if self._value is None:
            raise RuntimeError('No value set for {}'.format(self.flag))
        if self._value == '*': # If an asterisk is specified
            return [] # Just ignore the flag entirely
        return [self.flag, str(self._value)]

class RocBlasArgumentSet(cr.ArgumentSetABC):
    def _define_consistent_arguments(self):
        self.consistent_args['n'             ] = StripStarsArgument('-n'             )
        self.consistent_args['N'             ] = StripStarsArgument('-N'             )
        self.consistent_args['m'             ] = StripStarsArgument('-m'             )
        self.consistent_args['M'             ] = StripStarsArgument('-M'             )
        self.consistent_args['k'             ] = StripStarsArgument('-k'             )
        self.consistent_args['K'             ] = StripStarsArgument('-K'             )
        self.consistent_args['batch_count'   ] = StripStarsArgument('-b'             ) #
        self.consistent_args['function'      ] = StripStarsArgument('-f'             ) #
        self.consistent_args['compute_type'  ] = StripStarsArgument('-p'             ) # precision
        self.consistent_args['incx'          ] = StripStarsArgument('--incx'         )
        self.consistent_args['incy'          ] = StripStarsArgument('--incy'         )
        self.consistent_args['alpha'         ] = StripStarsArgument('--alpha'        )
        self.consistent_args['beta'          ] = StripStarsArgument('--beta'         )
        self.consistent_args['iters'         ] = StripStarsArgument('-i'             ) #
        self.consistent_args['samples'       ] = StripStarsArgument('-a'             )
        self.consistent_args['lda'           ] = StripStarsArgument('--lda'          )
        self.consistent_args['ldb'           ] = StripStarsArgument('--ldb'          )
        self.consistent_args['ldc'           ] = StripStarsArgument('--ldc'          )
        self.consistent_args['LDA'           ] = StripStarsArgument('--LDA'          )
        self.consistent_args['LDB'           ] = StripStarsArgument('--LDB'          )
        self.consistent_args['LDC'           ] = StripStarsArgument('--LDC'          )
        self.consistent_args['transA'        ] = StripStarsArgument('--transA'       )
        self.consistent_args['transB'        ] = StripStarsArgument('--transB'       )
        #self.consistent_args['initialization'] = StripStarsArgument('-initialization') # Unused?
        self.consistent_args['step_size'     ] = StripStarsArgument('-s'             )
        self.consistent_args['step_mult'     ] = cr.OptionalFlagArgument('-x'         ) # optional flag argument
        self.consistent_args['side'          ] = StripStarsArgument('--side'         )
        self.consistent_args['uplo'          ] = StripStarsArgument('--uplo'         )
        self.consistent_args['diag'          ] = StripStarsArgument('--diag'         )

    def _define_variable_arguments(self):
        #self.variable_args['nsample'] = cr.RequiredArgument('-i')
        self.variable_args['idir'] = cr.RequiredArgument('-w')
        self.variable_args['output_file'] = cr.RequiredArgument('-o')

    def __init__(self, **kwargs):
        cr.ArgumentSetABC.__init__(
                self, **kwargs
                )

    def get_full_command(self, run_configuration):
        timingscript = './timing.py'
        if not os.path.exists(timingscript):
            timingscript = os.path.join(os.path.dirname(os.path.realpath(__file__)), timingscript)
        else:
            timingscript = os.path.abspath(timingscript)
        if not os.path.exists(timingscript):
            raise RuntimeError('Unable to find {}!'.format(timingscript))

        #self.set('nsample', run_configuration.num_runs)
        self.set('idir', run_configuration.executable_directory)
        self.set('output_file', self.get_output_file(run_configuration))

        return [timingscript] + self.get_args()

    def collect_timing(self, run_configuration, data_type='time'):
        output_filename = self.get_output_file(run_configuration)
        rv = {}
        print('Processing {}'.format(output_filename))
        if os.path.exists(output_filename):
            # The output format is not consistent enough to justify using an out of the box reader.
            with open(output_filename, 'r') as raw_tsv:
                for line in raw_tsv.readlines():
                    # remove comment by splittling on `#` and taking the first segment
                    stripped_line = line.split('#')[0].strip()
                    if stripped_line:
                        split_line = stripped_line.split()
                        tag = int(split_line[0])
                        # Each line has 3 sections of num_samples, followed by the samples
                        samples = {}
                        read_idx = 1
                        for sample_type in ['time', 'gflops', 'bandwidth']:
                            num_samples = int(split_line[read_idx])
                            read_idx += 1
                            samples[sample_type] = [float(x) for x in split_line[read_idx:num_samples+read_idx]]
                            read_idx += num_samples
                        rv[tag] = samples[data_type]
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
        print(defaults)

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

class RocBlasYamlComparison(cr.Comparison):
    def __init__(self, data_type='time', **kwargs):
        cr.Comparison.__init__(self, **kwargs)
        self.data_type = data_type

    def plot(self, run_configurations, axes):
        label_map = OrderedDict()
        xlengths_map = OrderedDict()
        samples_map = OrderedDict()
        # Combine equivalent run configurations
        for run_configuration in run_configurations:
            for argument_set in self.argument_sets:
                key = run_configuration.get_id() + argument_set.get_hash()
                xlengths = xlengths_map[key] if key in xlengths_map else []
                samples = samples_map[key] if key in samples_map else []
                timing = argument_set.collect_timing(run_configuration,
                                                     data_type = self.data_type)
                for xlength, subsamples in timing.items():
                    for sample in subsamples:
                        xlengths.append(xlength)
                        samples.append(sample)
                if len(samples) > 0:
                    label_map[key] = run_configuration.label
                    xlengths_map[key] = xlengths
                    samples_map[key] = samples
        for key in label_map:
            axes.loglog(xlengths_map[key], samples_map[key], '.',
                        label = label_map[key],
                        markersize = 3,
                        )
        axes.set_xlabel('x-length (integer)')
        if self.data_type == 'time':
            axes.set_ylabel('Time (s)')
        if self.data_type == 'gflops':
            axes.set_ylabel('Speed (GFlops/s)')
        if self.data_type == 'bandwidth':
            axes.set_ylabel('Bandwidth (GB/s)')
        return len(label_map) > 0

data_type_classes = {}
class TimeComparison(RocBlasYamlComparison):
    def __init__(self, **kwargs):
        RocBlasYamlComparison.__init__(self, data_type='time', **kwargs)
data_type_classes['time'] = TimeComparison

class FlopsComparison(RocBlasYamlComparison):
    def __init__(self, **kwargs):
        RocBlasYamlComparison.__init__(self, data_type='gflops', **kwargs)
data_type_classes['gflops'] = FlopsComparison

class BandwidthComparison(RocBlasYamlComparison):
    def __init__(self, **kwargs):
        RocBlasYamlComparison.__init__(self, data_type='bandwidth', **kwargs)
data_type_classes['bandwidth'] = BandwidthComparison


class RocFftRunConfiguration(cr.RunConfiguration):
    def __init__(self, user_args, *args, **kwargs):
        cr.RunConfiguration.__init__(self, user_args, *args, **kwargs)


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

    comparisons = []

    #setup tests sorted by their respective figures
    for tests in data.test_cases:
        name = get_function_prefix(tests[0]['compute_type']) + tests[0]['function'].split('_')[0] + ' Performance'
        for data_type in user_args.data_types:
            data_type_cls = data_type_classes[data_type]
            comparison = data_type_cls(description=name)
            for test in tests:
                comparison.add( RocBlasArgumentSet(**{key:USED_YAML_KEYS_FMT[key](test[key]) for key in test if key in USED_YAML_KEYS_FMT}) )
            comparisons.append(comparison)

    command_runner.add_comparisons(comparisons)

    command_runner.execute()

    command_runner.show_plots()
    command_runner.output_summary()
